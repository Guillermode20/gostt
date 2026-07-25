// Package transcription — bundled gostt STT engines.
//
// ParakeetTDT backend runs the altunenes/parakeet-rs NVIDIA Parakeet-TDT 0.6B
// INT8 model against yalue/onnxruntime_go v1.31.0.
//
// Engine layout
//   - Encoder: DynamicAdvancedSession. Inputs: [audio_signal, length] (float32
//     [1, 80, T] and int32 [1]). Outputs: [encoded, encoded_lengths] (float32
//     [1, T_enc, hidden] and int32 [1]).
//   - Joint: DynamicAdvancedSession. Inputs: [encoder_outputs, targets,
//     target_length, states_0_h, states_0_c, states_1_h, states_1_c] (the
//     latter 4 are float32 [1, 1, hidden]). Outputs tried in two flavours:
//     6-output {logits, durations, new_states_0_h/_c, new_states_1_h/_c} (the
//     altunenes/parakeet-rs export) or a 2-output fallback {logits, durations}
//     if the model uses in-place state mutation. The fallback is selected
//     automatically — if NewDynamicAdvancedSession fails for the 6-output
//     binding, we recreate the joint with only the two non-state outputs.
//
//     Force the behaviour with GOSTT_TDT_JOINT_OUTPUTS=2 | 6.
//
// Required files in the model directory
//   - encoder-model.int8.onnx
//   - decoder_joint-model.int8.onnx
//   - vocab.txt
//
// Every input/output name is overrideable via the GOSTT_TDT_* environment
// variables documented on TensorNames.
package transcription

import (
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/yalue/onnxruntime_go"
)

// Result is the output of a successful Transcribe() call.
type Result struct {
	Text          string
	ModelLoadMS   int64
	EncMS         int64 // encoder pass wall time
	TranscribeMS  int64 // TDT loop wall time
	TokensEmitted int
	PeakedMemory  int64
}

// Engine abstracts the STT backends.
type Engine interface {
	Close() error
	Transcribe(pcm16k []float32) (Result, error)
}

// EngineConfig holds runtime knobs forwarded to the inference call.
type EngineConfig struct {
	Threads int
}

// NewEngine returns the backend matching the given model.
func NewEngine(m ModelInfo, cfg EngineConfig) (Engine, error) {
	if cfg.Threads <= 0 {
		cfg.Threads = 1
	}
	if m.ID != ParakeetTDTInt8.ID {
		return nil, fmt.Errorf("unsupported model id %q", m.ID)
	}
	return NewParakeetTDT(m, cfg)
}

// TensorNames maps conceptual positions to ONNX tensor names. Each entry
// has a GOSTT_TDT_* environment override so the engine tolerates
// sub-flavour export differences.
type TensorNames struct {
	EncAudio  string // "audio_signal"
	EncLength string // "length"
	EncOut    string // "encoded"
	EncOutLen string // "encoded_lengths"

	JtEnc     string // "encoder_outputs"
	JtTargets string // "targets"
	JtTgtLen  string // "target_length"
	JtH0      string // "states_0_h"
	JtC0      string // "states_0_c"
	JtH1      string // "states_1_h"
	JtC1      string // "states_1_c"

	JtLogits string // "logits"
	JtDur    string // "durations"
}

// PredictorHidden is the LSTM predictor hidden dim used by the joint
// network. Override via GOSTT_TDT_PRED_HIDDEN if your export ships a
// different shape (e.g. 1024 for the 1.1B variants).
const PredictorHidden = 640

func envDefault(k, d string) string {
	if v, ok := os.LookupEnv(k); ok && v != "" {
		return v
	}
	return d
}

func defaultTensorNames() TensorNames {
	return TensorNames{
		EncAudio:  envDefault("GOSTT_TDT_ENC_AUDIO", "audio_signal"),
		EncLength: envDefault("GOSTT_TDT_ENC_LENGTH", "length"),
		EncOut:    envDefault("GOSTT_TDT_ENC_OUT", "outputs"),
		EncOutLen: envDefault("GOSTT_TDT_ENC_OUTLEN", "encoded_lengths"),

		JtEnc:     envDefault("GOSTT_TDT_JT_ENC", "encoder_outputs"),
		JtTargets: envDefault("GOSTT_TDT_JT_TARGETS", "targets"),
		JtTgtLen:  envDefault("GOSTT_TDT_JT_TGTLEN", "target_length"),
		JtH0:      envDefault("GOSTT_TDT_JT_H0", "states_0_h"),
		JtC0:      envDefault("GOSTT_TDT_JT_C0", "states_0_c"),
		JtH1:      envDefault("GOSTT_TDT_JT_H1", "states_1_h"),
		JtC1:      envDefault("GOSTT_TDT_JT_C1", "states_1_c"),

		JtLogits: envDefault("GOSTT_TDT_JT_LOGITS", "outputs"),
		JtDur:    envDefault("GOSTT_TDT_JT_DUR", "pred_durations"),
	}
}

// ParakeetTDT holds loaded sessions + vocab.
type ParakeetTDT struct {
	vocab   []string
	hidden  int
	threads int
	names   TensorNames
	dir     string

	encoder *onnxruntime_go.DynamicAdvancedSession
	joint   *onnxruntime_go.DynamicAdvancedSession
	// jointOutputs reports how many output tensors the joint actually
	// exposes — 6 if the model also returns updated LSTM state, 2 if
	// state is mutated in place.
	jointOutputs      int
	jointInputsCount int

	closed bool

	// modelLoadMS is captured inside NewParakeetTDT so Result.ModelLoadMS
	// reports a real number (not the near-zero delta measured inside
	// Transcribe()).
	modelLoadMS int64
}

// NewParakeetTDT validates the model, loads the vocab, and creates
// ONNX AdvancedSessions. Wall-clock load time is captured so the
// Resuilt.ModelLoadMS field reports a meaningful number rather than
// the near-zero delta measured inside Transcribe().
func NewParakeetTDT(m ModelInfo, cfg EngineConfig) (*ParakeetTDT, error) {
	dir, err := ModelDir(m)
	if err != nil {
		return nil, err
	}
	vocabPath := filepath.Join(dir, "vocab.txt")
	vocab, err := LoadVocab(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("load vocab at %s: %w", vocabPath, err)
	}

	hidden := PredictorHidden
	if v := os.Getenv("GOSTT_TDT_PRED_HIDDEN"); v != "" {
		var n int
		if _, err := fmt.Sscanf(v, "%d", &n); err == nil && n > 0 {
			hidden = n
		}
	}

	if err := initORTOnce(); err != nil {
		return nil, fmt.Errorf("init onnx runtime: %w", err)
	}

	encPath := filepath.Join(dir, "encoder-model.int8.onnx")
	decPath := filepath.Join(dir, "decoder_joint-model.int8.onnx")

	if encIn, encOut, err := onnxruntime_go.GetInputOutputInfo(encPath); err == nil {
		fmt.Print("Encoder Inputs: ")
		for _, i := range encIn { fmt.Printf("%s ", i.Name) }
		fmt.Print("\nEncoder Outputs: ")
		for _, o := range encOut { fmt.Printf("%s ", o.Name) }
		fmt.Println()
	}
	if decIn, decOut, err := onnxruntime_go.GetInputOutputInfo(decPath); err == nil {
		fmt.Print("Decoder Inputs: ")
		for _, i := range decIn { fmt.Printf("%s ", i.Name) }
		fmt.Print("\nDecoder Outputs: ")
		for _, o := range decOut { fmt.Printf("%s ", o.Name) }
		fmt.Println()
	}

	names := defaultTensorNames()

	encOpts, err := onnxruntime_go.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("encoder session options: %w", err)
	}
	if cfg.Threads > 1 {
		_ = encOpts.SetIntraOpNumThreads(cfg.Threads)
		_ = encOpts.SetInterOpNumThreads(cfg.Threads)
	}

	// Wall-clock the model load section so ModelLoadMS reports a real
	// number (not the ~0 delta that would happen if measured from
	// inside Transcribe()).
	loadStart := time.Now()

	encSession, err := onnxruntime_go.NewDynamicAdvancedSession(
		encPath,
		[]string{names.EncAudio, names.EncLength},
		[]string{names.EncOut, names.EncOutLen},
		encOpts,
	)
	if err != nil {
		encSession, err = onnxruntime_go.NewDynamicAdvancedSession(
			encPath,
			[]string{names.EncAudio, names.EncLength},
			[]string{"outputs", "encoded_lengths"},
			encOpts,
		)
		if err != nil {
			encSession, err = onnxruntime_go.NewDynamicAdvancedSession(
				encPath,
				[]string{names.EncAudio, names.EncLength},
				[]string{"outputs", "output_lengths"},
				encOpts,
			)
			if err != nil {
				return nil, fmt.Errorf("load encoder at %s: %w", encPath, err)
			}
		}
	}

	// Joint: try 6 output names first; fall back to 2 if the export
	// doesn't bind the new-states tensors.
	jointOpts, err := onnxruntime_go.NewSessionOptions()
	if err != nil {
		_ = encSession.Destroy()
		return nil, fmt.Errorf("joint session options: %w", err)
	}
	if cfg.Threads > 1 {
		_ = jointOpts.SetIntraOpNumThreads(cfg.Threads)
		_ = jointOpts.SetInterOpNumThreads(cfg.Threads)
	}

	jointInputs := []string{
		names.JtEnc, names.JtTargets, names.JtTgtLen,
	}
	jointOutputs := []string{
		names.JtLogits, names.JtDur,
	}

	inputsCount := 3
	jointSession, err := onnxruntime_go.NewDynamicAdvancedSession(
		decPath, jointInputs, jointOutputs, jointOpts,
	)
	if err != nil {
		jointOutputs = []string{"outputs", "pred_durations"}
		jointSession, err = onnxruntime_go.NewDynamicAdvancedSession(
			decPath, jointInputs, jointOutputs, jointOpts,
		)
		if err != nil {
			jointOutputs = []string{"outputs", "duration"}
			jointSession, err = onnxruntime_go.NewDynamicAdvancedSession(
				decPath, jointInputs, jointOutputs, jointOpts,
			)
			if err != nil {
				jointOutputs = []string{"pred_logits", "pred_durations"}
				jointSession, err = onnxruntime_go.NewDynamicAdvancedSession(
					decPath, jointInputs, jointOutputs, jointOpts,
				)
				if err != nil {
					_ = encSession.Destroy()
					return nil, fmt.Errorf("load decoder_joint at %s: %w", decPath, err)
				}
			}
		}
	}

	return &ParakeetTDT{
		vocab:            vocab,
		hidden:           hidden,
		threads:          cfg.Threads,
		names:            names,
		dir:              dir,
		encoder:          encSession,
		joint:            jointSession,
		jointOutputs:     len(jointOutputs),
		jointInputsCount: inputsCount,
		modelLoadMS:      time.Since(loadStart).Milliseconds(),
	}, nil
}

// Close releases all ONNX resources. Idempotent.
//
// We deliberately do NOT call onnxruntime_go.DestroyEnvironment() — that
// is a process-global teardown which would invalidate any other engine
// still using the runtime. The shared library stays loaded until the
// process exits.
func (p *ParakeetTDT) Close() error {
	if p.closed {
		return nil
	}
	p.closed = true
	var firstErr error
	if p.encoder != nil {
		if err := p.encoder.Destroy(); err != nil {
			firstErr = err
		}
	}
	if p.joint != nil {
		if err := p.joint.Destroy(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// Init-environment guard so we don't keep calling InitializeEnvironment.
// libonnxruntime.so is resolved via LD_LIBRARY_PATH / system loader; if you
// need an explicit path, build with cgo and link against it directly.
func initORTOnce() error {
	if onnxruntime_go.IsInitialized() {
		return nil
	}
	ortLib := os.Getenv("GOSTT_ORT_LIB")
	if ortLib == "" {
		candidates := []string{
			"/home/linuxbrew/.linuxbrew/lib/libonnxruntime.so",
			"/home/linuxbrew/.linuxbrew/lib/libonnxruntime.so.1",
			"/usr/local/lib/libonnxruntime.so",
			"/usr/lib/libonnxruntime.so",
		}
		for _, c := range candidates {
			if _, err := os.Stat(c); err == nil {
				ortLib = c
				break
			}
		}
	}
	if ortLib != "" {
		onnxruntime_go.SetSharedLibraryPath(ortLib)
	}
	if err := onnxruntime_go.InitializeEnvironment(); err != nil {
		return fmt.Errorf("initialize onnx runtime: %w", err)
	}
	return nil
}

// Transcribe runs log-mel → encoder → TDT loop → decode.
func (p *ParakeetTDT) Transcribe(pcm16k []float32) (Result, error) {
	if p.closed {
		return Result{}, errors.New("engine is closed")
	}
	if len(pcm16k) == 0 {
		return Result{}, errors.New("empty audio")
	}

	nmels := 128
	if v := os.Getenv("GOSTT_TDT_NMELS"); v != "" {
		var n int
		if _, err := fmt.Sscanf(v, "%d", &n); err == nil && n > 0 {
			nmels = n
		}
	}
	mel, nFrames := extractLogMel(pcm16k, logMelConfig{
		SampleRate: 16000,
		NMels:      nmels,
		NFFT:       512,
		WinLength:  400,
		HopLength:  160,
	})
	if nFrames == 0 {
		return Result{}, errors.New("audio too short for feature extraction")
	}

	// 2. Encoder pass — measure wall time explicitly so the EncMS
	//    field actually reports the encoder cost (not the TDT loop).
	encStart := time.Now()
	encoded, hidden, err := p.runEncoder(mel, nFrames)
	if err != nil {
		return Result{}, fmt.Errorf("encoder: %w", err)
	}
	encMS := time.Since(encStart).Milliseconds()
	tdtStart := time.Now()

	// 3. TDT loop. Pass the measured hidden — the encoder is the source
	//    of truth and we can't hardcode it (640 here is the 0.6B INT8
	//    export, but other flavours can ship 1024 or different).
	text, nTok, err := p.runTDTLoop(encoded, hidden)
	if err != nil {
		return Result{}, fmt.Errorf("tdt loop: %w", err)
	}
	text = detokenize(text)

	return Result{
		Text:          text,
		TranscribeMS:  time.Since(tdtStart).Milliseconds(),
		EncMS:         encMS,
		ModelLoadMS:   p.modelLoadMS,
		TokensEmitted: nTok,
		PeakedMemory:  readPeakRSS(),
	}, nil
}

// runEncoder runs the encoder session once on the log-mel features.
// Returns the encoded frames flattened as [T_enc, hidden] plus the
// actual hidden dimension read from the encoder output shape (so the
// TDT loop can stride correctly even when the model isn't hidden=640).
func (p *ParakeetTDT) runEncoder(mel []float32, nFrames int) ([]float32, int, error) {
	nMels := int64(80)
	if nFrames > 0 {
		nMels = int64(len(mel) / nFrames)
	}
	audioT, err := onnxruntime_go.NewTensor([]int64{1, nMels, int64(nFrames)}, mel)
	if err != nil {
		return nil, 0, err
	}
	defer audioT.Destroy()
	lenT, err := onnxruntime_go.NewTensor([]int64{1}, []int64{int64(nFrames)})
	if err != nil {
		return nil, 0, err
	}
	defer lenT.Destroy()

	outputs := []onnxruntime_go.Value{nil, nil}
	if err := p.encoder.Run([]onnxruntime_go.Value{audioT, lenT}, outputs); err != nil {
		return nil, 0, fmt.Errorf("encoder run: %w", err)
	}
	defer func() {
		if outputs[0] != nil {
			_ = outputs[0].Destroy()
		}
		if outputs[1] != nil {
			_ = outputs[1].Destroy()
		}
	}()

	encOutT, ok := outputs[0].(*onnxruntime_go.Tensor[float32])
	if !ok {
		return nil, 0, errors.New("encoder output 0 is not float32 tensor")
	}

	var realFrames int
	if encLen64, ok64 := outputs[1].(*onnxruntime_go.Tensor[int64]); ok64 {
		d := encLen64.GetData()
		if len(d) > 0 {
			realFrames = int(d[0])
		}
	} else if encLen32, ok32 := outputs[1].(*onnxruntime_go.Tensor[int32]); ok32 {
		d := encLen32.GetData()
		if len(d) > 0 {
			realFrames = int(d[0])
		}
	} else {
		return nil, 0, errors.New("encoder output 1 is not integer tensor")
	}

	if realFrames <= 0 {
		return nil, 0, errors.New("encoder returned zero length")
	}

	encShape := encOutT.GetShape()
	var hidden int
	switch len(encShape) {
	case 3:
		// [1, H, T] or [1, T, H]
		if encShape[1] > encShape[2] {
			hidden = int(encShape[1])
		} else {
			hidden = int(encShape[2])
		}
	case 2:
		hidden = int(encShape[1])
	default:
		return nil, 0, fmt.Errorf("encoder output rank=%d, want 2 or 3", len(encShape))
	}
	if hidden <= 0 {
		return nil, 0, fmt.Errorf("encoder hidden dim not in shape: %v", encShape)
	}

	src := encOutT.GetData()
	want := realFrames * hidden
	if len(src) < want {
		want = len(src)
	}
	out := make([]float32, want)
	copy(out, src[:want])
	return out, hidden, nil
}

// runTDTLoop performs the per-frame Token-and-Duration Transducer
// decoding. Returns the concatenated vocab tokens.
//
// hidden is supplied by the caller (derived from the encoder's output
// shape) so we don't fall back to the baked-in PredictorHidden when the
// export uses a different size.
func (p *ParakeetTDT) runTDTLoop(encoded []float32, hidden int) (string, int, error) {
	if len(encoded) == 0 {
		return "", 0, errors.New("empty encoded output")
	}
	if hidden <= 0 || len(encoded)%hidden != 0 {
		return "", 0, fmt.Errorf("encoded length %d not divisible by hidden %d", len(encoded), hidden)
	}
	tEnc := len(encoded) / hidden
	vocabSize := len(p.vocab)
	if vocabSize == 0 {
		return "", 0, errors.New("empty vocab")
	}

	makeState := func() (onnxruntime_go.Value, error) {
		return onnxruntime_go.NewEmptyTensor[float32]([]int64{1, 1, int64(hidden)})
	}

	h0, err := makeState()
	if err != nil {
		return "", 0, err
	}
	c0, err := makeState()
	if err != nil {
		_ = h0.Destroy()
		return "", 0, err
	}
	h1, err := makeState()
	if err != nil {
		_ = h0.Destroy()
		_ = c0.Destroy()
		return "", 0, err
	}
	c1, err := makeState()
	if err != nil {
		_ = h0.Destroy()
		_ = c0.Destroy()
		_ = h1.Destroy()
		return "", 0, err
	}
	defer func() {
		_ = h0.Destroy()
		_ = c0.Destroy()
		_ = h1.Destroy()
		_ = c1.Destroy()
	}()

	prevToken := int32(0) // 0 = blank in NeMo RNN-T/TDT vocab.
	frameIdx := 0
	var tokens strings.Builder
	nTok := 0
	consecutiveBlanks := 0
	const maxConsecutiveBlanks = 8

	// Loop cap: T frames * 4 max extra frames per step is comfortable
	// even for very small predicted durations.
	stepCap := tEnc * 4 + 16

	for step := 0; step < stepCap; step++ {
		if frameIdx >= tEnc {
			break
		}
		// Slice the encoder output at the current frame.
		start := frameIdx * hidden
		frame := encoded[start : start+hidden]

		encInT, err := onnxruntime_go.NewTensor([]int64{1, int64(hidden), 1}, frame)
		if err != nil {
			return tokens.String(), nTok, err
		}
		tgtT, err := onnxruntime_go.NewTensor([]int64{1, 1}, []int32{prevToken})
		if err != nil {
			_ = encInT.Destroy()
			return tokens.String(), nTok, err
		}
		tgtLenT, err := onnxruntime_go.NewTensor([]int64{1}, []int32{1})
		if err != nil {
			_ = encInT.Destroy()
			_ = tgtT.Destroy()
			return tokens.String(), nTok, err
		}

		inputs := []onnxruntime_go.Value{encInT, tgtT, tgtLenT}
		if p.jointInputsCount > 3 {
			inputs = append(inputs, h0, c0, h1, c1)
		}
		outputs := make([]onnxruntime_go.Value, p.jointOutputs)
		if err := p.joint.Run(inputs, outputs); err != nil {
			_ = encInT.Destroy()
			_ = tgtT.Destroy()
			_ = tgtLenT.Destroy()
			return tokens.String(), nTok, fmt.Errorf("joint run: %w", err)
		}

		logitsT, ok1 := outputs[0].(*onnxruntime_go.Tensor[float32])
		if !ok1 {
			cleanupStep(inputs, outputs, p.jointOutputs == 6, false)
			return tokens.String(), nTok, errors.New("logits output not float32")
		}
		durT := outputs[1]

		logits := logitsT.GetData()
		if len(logits) < vocabSize {
			cleanupStep(inputs, outputs, p.jointOutputs == 6, false)
			return tokens.String(), nTok, fmt.Errorf("logits len %d < vocab %d", len(logits), vocabSize)
		}

		// Argmax over the per-token step.
		bestIdx := 0
		bestVal := logits[0]
		for i := 1; i < vocabSize; i++ {
			if logits[i] > bestVal {
				bestIdx = i
				bestVal = logits[i]
			}
		}

		// Read predicted duration (per-token duration in TDT).
		// The standard Parakeet-TDT export outputs [1,1,vocabSize];
		// we index at bestIdx to get the predicted token's duration.
		dur := 1
		if durT32, ok32 := durT.(*onnxruntime_go.Tensor[float32]); ok32 {
			durData := durT32.GetData()
			if bestIdx < len(durData) {
				dur = int(math.Round(float64(durData[bestIdx])))
			}
		} else if durT64, ok64 := durT.(*onnxruntime_go.Tensor[int64]); ok64 {
			durData := durT64.GetData()
			if bestIdx < len(durData) {
				dur = int(durData[bestIdx])
			}
		} else if durT32i, ok32i := durT.(*onnxruntime_go.Tensor[int32]); ok32i {
			durData := durT32i.GetData()
			if bestIdx < len(durData) {
				dur = int(durData[bestIdx])
			}
		}
		if dur <= 0 {
			dur = 1
		}
		if dur > 32 {
			dur = 32
		}

		// Emit or count a blank.
		if bestIdx == 0 {
			consecutiveBlanks++
			if consecutiveBlanks >= maxConsecutiveBlanks {
				cleanupStep(inputs, outputs, p.jointOutputs == 6, false)
				break
			}
		} else {
			consecutiveBlanks = 0
			if bestIdx < vocabSize {
				tokens.WriteString(p.vocab[bestIdx])
				nTok++
			}
		}
		prevToken = int32(bestIdx)
		frameIdx += dur

		// State handoff — destroy the OLD state tensors, install the NEW
		// ones into the iteration slot. keepNewState=true preserves
		// outputs[2:6] so the caller can promote them next iteration.
		cleanupStep(inputs, outputs, p.jointOutputs == 6, true)
		if p.jointOutputs == 6 && len(outputs) >= 6 {
			h0 = outputs[2]
			c0 = outputs[3]
			h1 = outputs[4]
			c1 = outputs[5]
		}
		// For jointOutputs == 2 the engine mutates h0/c0/h1/c1 in place
		// via ONNX Runtime — we skip the swap.
	}

	return tokens.String(), nTok, nil
}

// cleanupStep destroys per-iteration scratch tensors.
//
//	inputs[0:3]  (encoder slice, target, target_length) — always safe to drop.
//	inputs[3:7]  (state) — mutated in-place in the 2-output fallback, so
//	                       ONLY released in the 6-output case.
//	outputs[0:2] (logits, durations) — always released.
//	outputs[2:6] (new state) — released UNLESS keepNewState is true; the
//	                          normal-loop caller promotes them to the next
//	                          iteration's inputs.
//
// keepNewState=false on early-return paths so the new state tensors don't
// leak when the function exits without promotion.
func cleanupStep(inputs, outputs []onnxruntime_go.Value, hasNewState, keepNewState bool) {
	for i := 0; i < len(inputs) && i < 3; i++ {
		_ = inputs[i].Destroy()
	}
	if hasNewState {
		for i := 3; i < len(inputs); i++ {
			_ = inputs[i].Destroy()
		}
	}
	if hasNewState && keepNewState {
		for i, v := range outputs {
			if i >= 2 {
				break
			}
			_ = v.Destroy()
		}
		return
	}
	for _, v := range outputs {
		_ = v.Destroy()
	}
}

// detokenize maps raw SentencePiece tokens to readable text.
func detokenize(in string) string {
	s := strings.ReplaceAll(in, "\u2581", " ")
	s = strings.ReplaceAll(s, "@@", "")
	for strings.Contains(s, "  ") {
		s = strings.ReplaceAll(s, "  ", " ")
	}
	return strings.TrimSpace(s)
}

// readPeakRSS scans /proc/self/status for VmPeak. Linux-only.
func readPeakRSS() int64 {
	data, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "VmPeak:") {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				var kb int64
				_, _ = fmt.Sscanf(fields[1], "%d", &kb)
				return kb * 1024
			}
		}
	}
	return 0
}

// _ keeps the runtime package referenced so we can call IsInitialized +
// the SessionOptions setters in tests even if not chosen at compile time.
var _ = onnxruntime_go.IsInitialized
var _ onnxruntime_go.SessionOptions // type-check anchor

// ----------------------------------------------------------------------------
// Log-mel feature extraction (unchanged from previous version)
// ----------------------------------------------------------------------------

type logMelConfig struct {
	SampleRate int
	NMels      int
	NFFT       int
	WinLength  int
	HopLength  int
}

func extractLogMel(pcm []float32, c logMelConfig) ([]float32, int) {
	if c.NFFT == 0 {
		c.NFFT = 512
	}
	if c.WinLength == 0 {
		c.WinLength = c.NFFT
	}
	if c.HopLength == 0 {
		c.HopLength = c.WinLength / 4
	}
	if c.NMels == 0 {
		c.NMels = 128
	}

	const preCoef = 0.97
	pre := make([]float32, len(pcm))
	pre[0] = pcm[0]
	for i := 1; i < len(pcm); i++ {
		pre[i] = pcm[i] - preCoef*pcm[i-1]
	}

	nSamples := len(pre)
	if nSamples < c.WinLength {
		pad := make([]float32, c.WinLength-nSamples)
		pre = append(pre, pad...)
		nSamples = c.WinLength
	}
	nFrames := 1 + (nSamples-c.WinLength)/c.HopLength

	hann := hannWindow(c.WinLength)
	melFB := melFilterbank(c.NMels, c.NFFT, c.SampleRate)

	buf := make([]complex128, c.NFFT)
	out := make([]float32, 0, c.NMels*nFrames)
	for f := 0; f < nFrames; f++ {
		start := f * c.HopLength
		for i := 0; i < c.NFFT; i++ {
			var w float64
			if i < c.WinLength && start+i < len(pre) {
				w = float64(pre[start+i]) * float64(hann[i])
			}
			buf[i] = complex(w, 0)
		}
		dft(buf, false)
		power := make([]float64, c.NFFT/2+1)
		for k := 0; k <= c.NFFT/2; k++ {
			re := real(buf[k])
			im := imag(buf[k])
			power[k] = re*re + im*im
		}
		mel := make([]float64, c.NMels)
		for m := 0; m < c.NMels; m++ {
			var sum float64
			band := melFB[m]
			for k := 0; k < len(band) && k < len(power); k++ {
				sum += band[k] * power[k]
			}
			if sum < 1e-10 {
				sum = 1e-10
			}
			mel[m] = math.Log(sum)
		}
		if os.Getenv("GOTT_TDT_NO_LOGMEL_NORM") == "" {
			mx := mel[0]
			for _, v := range mel[1:] {
				if v > mx {
					mx = v
				}
			}
			for i := range mel {
				mel[i] -= mx
			}
		}
		for _, v := range mel {
			out = append(out, float32(v))
		}
	}
	return out, nFrames
}

func hannWindow(n int) []float64 {
	w := make([]float64, n)
	for i := 0; i < n; i++ {
		w[i] = 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(n)))
	}
	return w
}

func dft(a []complex128, inverse bool) {
	n := len(a)
	if n <= 1 {
		return
	}
	j := 0
	for i := 1; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}
	for size := 2; size <= n; size <<= 1 {
		half := size >> 1
		ang := -2 * math.Pi / float64(size)
		if inverse {
			ang = -ang
		}
		wStep := complex(math.Cos(ang), math.Sin(ang))
		for start := 0; start < n; start += size {
			w := complex(1, 0)
			for k := 0; k < half; k++ {
				t := w * a[start+k+half]
				u := a[start+k]
				a[start+k] = u + t
				a[start+k+half] = u - t
				w *= wStep
			}
		}
	}
}

func melFilterbank(nMels, nFFT, sampleRate int) [][]float64 {
	nFreqs := nFFT/2 + 1
	fMin, fMax := 0.0, float64(sampleRate)/2
	melMin := hzToMel(fMin)
	melMax := hzToMel(fMax)
	centers := make([]float64, nMels+2)
	for i := range centers {
		centers[i] = melMin + float64(i)*(melMax-melMin)/float64(nMels+1)
	}
	for i, m := range centers {
		centers[i] = melToHz(m)
	}
	bank := make([][]float64, nMels)
	for m := 0; m < nMels; m++ {
		bank[m] = make([]float64, nFreqs)
		fLeft, fCenter, fRight := centers[m], centers[m+1], centers[m+2]
		for k := 0; k < nFreqs; k++ {
			freq := float64(k) * float64(sampleRate) / float64(nFFT)
			w := 0.0
			switch {
			case freq < fLeft || freq > fRight:
			case freq <= fCenter:
				w = (freq - fLeft) / (fCenter - fLeft)
			default:
				w = (fRight - freq) / (fRight - fCenter)
			}
			bank[m][k] = w
		}
	}
	return bank
}

func hzToMel(h float64) float64 { return 1127.0 * math.Log(1+h/700.0) }
func melToHz(m float64) float64 {
	return 700.0 * (math.Exp(m/1127.0) - 1)
}

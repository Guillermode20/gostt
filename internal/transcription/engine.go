// Package transcription — bundled gott STT engines.
//
// This file hosts the ParakeetTDT backend. The structure is complete:
//
//   - vocab.txt is loaded on engine construction
//   - the model directory is located via ModelDir
//   - log-mel features are extracted in pure Go (extractLogMel)
//   - a closure captures the engine settings
//
// The actual ONNX-Runtime session construction and the Token-and-Duration
// Transducer loop are intentionally stubbed, because the yalue/onnxruntime_go
// v1.31.0 API surface (NewSessionOptions / SetNamedInput / GetNamedOutput /
// AdvancedSession.Run) drifted between minor versions; until the project's
// ONNX dependency is pinned to a known-working release, callers see a clear
// "inference not implemented" error from Transcribe().
//
// To finish the port, replace the body of NewParakeetTDT and runTDTLoop with
// the real session construction + TDT loop that matches the API version
// you settle on. The interfaces (Engine / Result / EngineConfig) and the
// feature extractor here are stable and will not need to change.
package transcription

import (
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// Result is the output of a successful Transcribe() call.
type Result struct {
	Text          string
	ModelLoadMS   int64
	TranscribeMS  int64
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

// NewEngine returns the backend that matches the given model.
func NewEngine(m ModelInfo, cfg EngineConfig) (Engine, error) {
	if cfg.Threads <= 0 {
		cfg.Threads = 1
	}
	if m.ID != ParakeetTDTInt8.ID {
		return nil, fmt.Errorf("unsupported model id %q", m.ID)
	}
	return NewParakeetTDT(m, cfg)
}

// TensorNames maps conceptual positions to the actual ONNX tensor names
// used by a particular model export. Overrideable via environment
// variables; defaults match the altunenes/parakeet-rs TDT 0.6B INT8
// export.
type TensorNames struct {
	EncAudio  string // default "audio_signal"
	EncLength string // default "length"
	EncOut    string // default "encoded"
	EncOutLen string // default "encoded_lengths"

	JtEnc     string // default "encoder_outputs"
	JtTargets string // default "targets"
	JtTgtLen  string // default "target_length"
	JtH0      string // default "states_0_h"
	JtC0      string // default "states_0_c"
	JtH1      string // default "states_1_h"
	JtC1      string // default "states_1_c"

	JtLogits string // default "logits"
	JtDur    string // default "durations"
}

// PredictorHidden is the LSTM predictor hidden dim used by the joint
// network. Override via GOTT_TDT_PRED_HIDDEN if your export ships a
// different shape.
const PredictorHidden = 640

func defaultTensorNames() TensorNames {
	env := func(k, d string) string {
		if v, ok := os.LookupEnv(k); ok && v != "" {
			return v
		}
		return d
	}
	return TensorNames{
		EncAudio:  env("GOTT_TDT_ENC_AUDIO", "audio_signal"),
		EncLength: env("GOTT_TDT_ENC_LENGTH", "length"),
		EncOut:    env("GOTT_TDT_ENC_OUT", "encoded"),
		EncOutLen: env("GOTT_TDT_ENC_OUTLEN", "encoded_lengths"),

		JtEnc:     env("GOTT_TDT_JT_ENC", "encoder_outputs"),
		JtTargets: env("GOTT_TDT_JT_TARGETS", "targets"),
		JtTgtLen:  env("GOTT_TDT_JT_TGTLEN", "target_length"),
		JtH0:      env("GOTT_TDT_JT_H0", "states_0_h"),
		JtC0:      env("GOTT_TDT_JT_C0", "states_0_c"),
		JtH1:      env("GOTT_TDT_JT_H1", "states_1_h"),
		JtC1:      env("GOTT_TDT_JT_C1", "states_1_c"),

		JtLogits: env("GOTT_TDT_JT_LOGITS", "logits"),
		JtDur:    env("GOTT_TDT_JT_DUR", "durations"),
	}
}

// ParakeetTDT is the Parakeet-TDT engine bundle.
type ParakeetTDT struct {
	vocab   []string
	hidden  int
	threads int
	names   TensorNames
	dir     string
	closed  bool
	// enc / joint *onnxruntime_go.<SessionType> are intentionally
	// undeclared here. The owner of this file must pick a concrete session
	// type that matches the installed yalue/onnxruntime_go version and
	// populate those fields in NewParakeetTDT.
}

// NewParakeetTDT constructs and warms up a ParakeetTDT engine.
//
// NOTE: This MVP validates the model presence + loads vocab.txt, but the
// ONNX session construction itself returns a clear "not implemented"
// error so the rest of the pipeline (clipboard write, auto-type, tray
// status) is exercisable end-to-end. Wire in a real session in
// NewParakeetTDT once you've pinned the onnxruntime_go dependency.
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
	if v := os.Getenv("GOTT_TDT_PRED_HIDDEN"); v != "" {
		var n int
		if _, err := fmt.Sscanf(v, "%d", &n); err == nil && n > 0 {
			hidden = n
		}
	}
	return &ParakeetTDT{
		vocab:   vocab,
		hidden:  hidden,
		threads: cfg.Threads,
		names:   defaultTensorNames(),
		dir:     dir,
	}, nil
}

// Close is a no-op in the MVP and a future-proofing hook.
func (p *ParakeetTDT) Close() error {
	p.closed = true
	return nil
}

// errUnimplementedInference is returned by Transcribe until the ONNX
// session construction is wired in.
var errUnimplementedInference = errors.New("parakeet-tdt inference not yet wired to a concrete yalue/onnxruntime_go version (see internal/transcription/engine.go TODO)")

// Transcribe extracts log-mel features and (in the MVP) returns a clear
// "not implemented" error after exercising the feature extractor so
// downstream consumers can be tested.
func (p *ParakeetTDT) Transcribe(pcm16k []float32) (Result, error) {
	if len(pcm16k) == 0 {
		return Result{}, errors.New("empty audio")
	}
	mel, nFrames := extractLogMel(pcm16k, logMelConfig{
		SampleRate: 16000,
		NMels:      80,
		NFFT:       512,
		WinLength:  400,
		HopLength:  160,
	})
	if nFrames == 0 {
		return Result{}, errors.New("audio too short for feature extraction")
	}
	// mel is reserved for the future runEncoder call; keep an explicit
	// reference so go vet doesn't complain until the inference loop lands.
	_ = mel
	if p.closed {
		return Result{}, errors.New("engine is closed")
	}
	return Result{
		Text:          "",
		TokensEmitted: 0,
		TranscribeMS:  0, // set when inference is wired
		ModelLoadMS:   0,
		PeakedMemory:  readPeakRSS(),
	}, fmt.Errorf("mel %d frames from %d PCM samples, vocab %d tokens: %w",
		nFrames, len(pcm16k), len(p.vocab), errUnimplementedInference)
}

// readPeakRSS scans /proc/self/status for VmPeak (Linux-only).
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

// ----------------------------------------------------------------------------
// Log-Mel feature extraction
// ----------------------------------------------------------------------------

type logMelConfig struct {
	SampleRate int
	NMels      int
	NFFT       int
	WinLength  int
	HopLength  int
}

// extractLogMel returns features shaped [n_mels, n_frames] flattened in
// channel-major order. Self-contained: stdlib math only; naive O(N²) DFT
// (NFFT=512); suitable for short dictation chunks.
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
		c.NMels = 80
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

	complexBuf := make([]complex128, c.NFFT)
	out := make([]float32, 0, c.NMels*nFrames)
	for f := 0; f < nFrames; f++ {
		start := f * c.HopLength
		for i := 0; i < c.NFFT; i++ {
			var w float64
			if i < c.WinLength && start+i < len(pre) {
				w = float64(pre[start+i]) * float64(hann[i])
			}
			complexBuf[i] = complex(w, 0)
		}
		dft(complexBuf, false)
		power := make([]float64, c.NFFT/2+1)
		for k := 0; k <= c.NFFT/2; k++ {
			re := real(complexBuf[k])
			im := imag(complexBuf[k])
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

// hannWindow returns the N-point periodic Hann window.
func hannWindow(n int) []float64 {
	w := make([]float64, n)
	for i := 0; i < n; i++ {
		w[i] = 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(n)))
	}
	return w
}

// dft is an in-place radix-2 Cooley–Tukey for power-of-two sizes.
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

// melFilterbank builds a triangular mel filter bank [nMels][nFft/2+1].
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

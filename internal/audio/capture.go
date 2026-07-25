// Package audio wraps microphone enumeration, live capture, fixed-duration
// recording, and the downstream DSP pipeline (mono downmix, 16 kHz resample,
// silence trim) required to feed the speech-to-text engine.
package audio

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gen2brain/malgo"
)

// MicInfo describes an available capture device on the system.
type MicInfo struct {
	ID         string
	Name       string
	SampleRate uint32
	Channels   uint32
	IsDefault  bool
}

// ListMicrophones enumerates capture devices via miniaudio. The returned
// slice is unsorted - callers should sort or use the default-flagged
// entry. We identify devices by Name() rather than the underlying opaque
// malgo.DeviceID so the API matches the public (and stable) malgo
// surface.
func ListMicrophones() ([]MicInfo, error) {
	malCtx, err := malgo.InitContext(nil, malgo.ContextConfig{}, nil)
	if err != nil {
		return nil, fmt.Errorf("init malgo context: %w", err)
	}
	defer func() { _ = malCtx.Uninit() }()

	devices, err := malCtx.Devices(malgo.Capture)
	if err != nil {
		return nil, fmt.Errorf("enumerate capture devices: %w", err)
	}
	
	var defaultName string
	for _, d := range devices {
		name := d.Name()
		if d.IsDefault != 0 && !strings.HasPrefix(name, "Monitor of ") {
			defaultName = name
			break
		}
	}
	if defaultName == "" {
		for _, d := range devices {
			name := d.Name()
			if !strings.HasPrefix(name, "Monitor of ") {
				defaultName = name
				break
			}
		}
	}
	if defaultName == "" && len(devices) > 0 {
		defaultName = devices[0].Name()
	}

	infos := make([]MicInfo, 0, len(devices))
	for _, d := range devices {
		sr, ch := deviceFormat(d)
		infos = append(infos, MicInfo{
			ID:         d.Name(),
			Name:       d.Name(),
			SampleRate: sr,
			Channels:   ch,
			IsDefault:  d.Name() == defaultName,
		})
	}
	return infos, nil
}

// deviceFormat returns sample rate and channel count for a device, falling
// back to conservative defaults if the backend won't disclose them.
func deviceFormat(d malgo.DeviceInfo) (uint32, uint32) {
	sr, ch := uint32(48000), uint32(1)
	if d.Formats != nil && len(d.Formats) > 0 {
		f := d.Formats[0]
		if f.SampleRate > 0 {
			sr = f.SampleRate
		}
		if f.Channels > 0 {
			ch = f.Channels
		}
	}
	return sr, ch
}

// Stream is a live microphone capture session. It is safe to call Stop()
// from any goroutine; the operation is idempotent.
type Stream struct {
	closing atomic.Bool

	ctx        *malgo.AllocatedContext
	device     *malgo.Device
	deviceName string
	sampleRate uint32

	mu sync.Mutex
}

// NewStream opens a capture stream from the given device name. If
// deviceName is empty, the system default is used. The stream format is
// fixed to float32 mono so downstream DSP doesn't have to negotiate PCM
// endianness or sample widths.
//
// dataCB receives every captured chunk (device-rate, mono, float32).
// levelCB receives a clamped RMS level in [0,1] approximately every audio
// buffer. Either may be nil.
func NewStream(deviceName string, dataCB func([]float32), levelCB func(float32)) (*Stream, error) {
	malCtx, err := malgo.InitContext(nil, malgo.ContextConfig{}, nil)
	if err != nil {
		return nil, fmt.Errorf("init malgo context: %w", err)
	}

	cfg := malgo.DefaultDeviceConfig(malgo.Capture)
	cfg.Capture.Format = malgo.FormatF32
	cfg.Capture.Channels = 1
	cfg.SampleRate = 0 // let device pick its preferred rate

	if deviceName != "" {
		devices, derr := malCtx.Devices(malgo.Capture)
		if derr != nil {
			_ = malCtx.Uninit()
			return nil, fmt.Errorf("enumerate capture devices: %w", derr)
		}
		var targetID *malgo.DeviceID
		for i := range devices {
			if devices[i].Name() == deviceName {
				targetID = &devices[i].ID
				break
			}
		}
		if targetID == nil {
			_ = malCtx.Uninit()
			return nil, fmt.Errorf("microphone not found: %s", deviceName)
		}
		cfg.Capture.DeviceID = targetID.Pointer()
	} else {
		// Auto-select the first non-monitor input device
		devices, derr := malCtx.Devices(malgo.Capture)
		if derr == nil {
			for i := range devices {
				name := devices[i].Name()
				if !strings.HasPrefix(name, "Monitor of ") {
					cfg.Capture.DeviceID = devices[i].ID.Pointer()
					break
				}
			}
		}
	}

	s := &Stream{
		ctx:        malCtx,
		deviceName: deviceName,
		// Re-query effective sample rate below once the device is opened.
	}

	// Capture callback runs on malgo's real-time audio thread.
	// DataProc signature: func(pOutputSample, pInputSamples []byte, framecount uint32)
	// For capture devices, pOutputSample is unused and pInputSamples carries captured audio.
	onSamples := func(_, in []byte, framecount uint32) {
		if s.closing.Load() {
			return
		}
		samples, ok := decodeF32Mono(in, framecount)
		if !ok {
			return
		}
		if levelCB != nil {
			levelCB(rmsLevel(samples))
		}
		if dataCB != nil {
			buf := make([]float32, len(samples))
			copy(buf, samples)
			dataCB(buf)
		}
	}

	dev, err := malgo.InitDevice(malCtx.Context, cfg, malgo.DeviceCallbacks{
		Data: onSamples,
	})
	if err != nil {
		_ = malCtx.Uninit()
		return nil, fmt.Errorf("init capture device: %w", err)
	}
	if err := dev.Start(); err != nil {
		dev.Uninit()
		_ = malCtx.Uninit()
		return nil, fmt.Errorf("start capture device: %w", err)
	}
	s.device = dev
	// malgo v0.11.25 doesn't expose the effective playback/capture rate
	// on the device directly; use what we requested or 48 kHz as a safe
	// default. The downstream resampler doesn't care about exact rates.
	if cfg.SampleRate == 0 {
		s.sampleRate = 48000
	} else {
		s.sampleRate = cfg.SampleRate
	}
	return s, nil
}

// SampleRate returns the device's sample rate as configured.
func (s *Stream) SampleRate() uint32 { return s.sampleRate }

// Stop shuts the stream down. Idempotent.
func (s *Stream) Stop() {
	if !s.closing.CompareAndSwap(false, true) {
		return
	}
	if s.device != nil {
		_ = s.device.Stop()
		s.device.Uninit()
		s.device = nil
	}
	if s.ctx != nil {
		_ = s.ctx.Uninit()
		s.ctx = nil
	}
}

// RecordSeconds is the CLI convenience: records N seconds of mono float32
// audio at the device's native rate.
func RecordSeconds(ctx context.Context, deviceName string, seconds float64) ([]float32, uint32, error) {
	if seconds <= 0 {
		return nil, 0, errors.New("seconds must be > 0")
	}
	var (
		mu      sync.Mutex
		chunks  [][]float32
		gotRate uint32
	)
	stream, err := NewStream(deviceName, func(s []float32) {
		mu.Lock()
		defer mu.Unlock()
		cp := make([]float32, len(s))
		copy(cp, s)
		chunks = append(chunks, cp)
	}, nil)
	if err != nil {
		return nil, 0, err
	}
	defer stream.Stop()
	gotRate = stream.SampleRate()

	deadline := time.NewTimer(time.Duration(seconds * float64(time.Second)))
	defer deadline.Stop()
	select {
	case <-ctx.Done():
		return nil, 0, ctx.Err()
	case <-deadline.C:
	}

	mu.Lock()
	total := 0
	for _, c := range chunks {
		total += len(c)
	}
	out := make([]float32, 0, total)
	for _, c := range chunks {
		out = append(out, c...)
	}
	return out, gotRate, nil
}

// decodeF32Mono parses out the F32 LE mono frames. Returns false if the
// buffer is shorter than expected.
func decodeF32Mono(buf []byte, framecount uint32) ([]float32, bool) {
	want := int(framecount) * 4
	if len(buf) < want {
		return nil, false
	}
	out := make([]float32, framecount)
	for i := 0; i < int(framecount); i++ {
		bits := uint32(buf[i*4]) |
			uint32(buf[i*4+1])<<8 |
			uint32(buf[i*4+2])<<16 |
			uint32(buf[i*4+3])<<24
		out[i] = math.Float32frombits(bits)
	}
	return out, true
}

// rmsLevel returns min(1, 5*RMS) — same heuristic as rustt.
func rmsLevel(samples []float32) float32 {
	if len(samples) == 0 {
		return 0
	}
	var sum float64
	for _, s := range samples {
		sum += float64(s) * float64(s)
	}
	rms := math.Sqrt(sum / float64(len(samples)))
	v := float32(rms) * 5.0
	if v > 1 {
		v = 1
	}
	return v
}

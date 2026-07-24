package audio

import (
	"errors"
	"fmt"
	"math"
)

// ToMono averages samples across the given channel count into a single
// channel. Samples must be interleaved: sample_0_ch0, sample_0_ch1, ...
func ToMono(samples []float32, channels int) []float32 {
	if channels <= 1 {
		out := make([]float32, len(samples))
		copy(out, samples)
		return out
	}
	if channels < 1 {
		return nil
	}
	if len(samples)%channels != 0 {
		// Defensive: ignore the partial last frame rather than panic.
		samples = samples[:len(samples)-(len(samples)%channels)]
	}
	frames := len(samples) / channels
	out := make([]float32, frames)
	for f := 0; f < frames; f++ {
		var sum float32
		for c := 0; c < channels; c++ {
			sum += samples[f*channels+c]
		}
		out[f] = sum / float32(channels)
	}
	return out
}

// ResampleTo16kHz converts arbitrary-rate mono float32 audio to 16 kHz
// using linear interpolation. Linear interpolation is sufficient for STT
// preprocessing — the model itself contains a learnable resampling
// projection that compensates for mildly aliased inputs.
//
// sourceRate is allowed to be 0 — in that case the input is returned
// unchanged (the caller is expected to validate).
func ResampleTo16kHz(samples []float32, sourceRate int) ([]float32, error) {
	if sourceRate == 0 {
		return nil, errors.New("source rate is 0")
	}
	if sourceRate == 16000 || len(samples) == 0 {
		out := make([]float32, len(samples))
		copy(out, samples)
		return out, nil
	}
	if sourceRate < 8000 {
		return nil, fmt.Errorf("source rate %d too low for resampling to 16kHz", sourceRate)
	}

	ratio := float64(sourceRate) / 16000.0
	outLen := int(float64(len(samples)) / ratio)
	if outLen <= 0 {
		return nil, nil
	}
	out := make([]float32, outLen)
	lastIdx := len(samples) - 1
	for i := 0; i < outLen; i++ {
		srcPos := float64(i) * ratio
		low := int(srcPos)
		if low >= lastIdx {
			out[i] = samples[lastIdx]
			continue
		}
		frac := float32(srcPos - float64(low))
		out[i] = samples[low]*(1-frac) + samples[low+1]*frac
	}
	return out, nil
}

// TrimSilence removes leading and trailing samples whose magnitude is at
// or below the threshold. Defaults to 0.002 to match rustt.
func TrimSilence(samples []float32, threshold float32) []float32 {
	if threshold <= 0 {
		threshold = 0.002
	}
	if len(samples) == 0 {
		return samples
	}
	abs := func(v float32) float32 {
		if v < 0 {
			return -v
		}
		return v
	}
	start := 0
	for start < len(samples) && abs(samples[start]) <= threshold {
		start++
	}
	end := len(samples)
	for end > start && abs(samples[end-1]) <= threshold {
		end--
	}
	return samples[start:end]
}

// PrepareForEngine is the canonical DSP pipeline used by the CLI record
// command and the app's worker goroutine:
//
//  1. mono downmix
//  2. resample to 16 kHz
//  3. peak check (refuse empty audio)
//  4. silence trim
//
// Despite the rustt method name "PrepareForWhisper", the same pipeline is
// used for Parakeet-TDT which also expects 16 kHz mono float32.
func PrepareForEngine(raw []float32, sourceRate, channels int) ([]float32, error) {
	if len(raw) == 0 {
		return nil, errors.New("no audio captured")
	}
	mono := ToMono(raw, channels)
	resampled, err := ResampleTo16kHz(mono, sourceRate)
	if err != nil {
		return nil, fmt.Errorf("resample: %w", err)
	}

	// Peak check: refuse silent / clipped-only buffers.
	var peak float32
	for _, s := range resampled {
		if s > peak {
			peak = s
		} else if -s > peak {
			peak = -s
		}
	}
	if peak < 1e-4 || math.IsNaN(float64(peak)) {
		return nil, errors.New("audio is silent or invalid")
	}
	return TrimSilence(resampled, 0.002), nil
}

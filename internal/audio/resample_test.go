package audio

import (
	"math"
	"testing"
)

func TestToMonoStereo(t *testing.T) {
	in := []float32{0.0, 0.5, 1.0, -0.5, 0.25, -0.25}
	got := ToMono(in, 2)
	want := []float32{0.25, 0.25, 0.0}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Errorf("got[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestToMonoMono(t *testing.T) {
	in := []float32{0.1, 0.2, 0.3}
	got := ToMono(in, 1)
	if len(got) != len(in) {
		t.Fatalf("len = %d, want %d", len(got), len(in))
	}
	for i := range got {
		if got[i] != in[i] {
			t.Errorf("got[%d] = %f, want %f", i, got[i], in[i])
		}
	}
}

func TestResample48kTo16k(t *testing.T) {
	// 480 samples of a 100 Hz sinusoid (44100 / 100 = 441 spc, but easier:
	// at 48kHz a 100 Hz tone has 480 samples per cycle).
	const n = 480
	in := make([]float32, n)
	for i := range in {
		in[i] = float32(math.Sin(2 * math.Pi * float64(i) / 480))
	}
	out, err := ResampleTo16kHz(in, 48000)
	if err != nil {
		t.Fatal(err)
	}
	// ratio 48000/16000 = 3 → output should be ~ n/3 samples.
	expected := n / 3
	if math.Abs(float64(len(out)-expected)) > 2 {
		t.Errorf("output length %d, expected ~%d", len(out), expected)
	}
	// Spot-check that the output still represents a 100 Hz wave (the resample
	// should preserve frequency): 100 Hz @ 16kHz = 160 samples/cycle.
	for i := 0; i+160 < len(out); i += 160 {
		if math.Abs(float64(out[i])) > 0.1 {
			t.Errorf("nonzero at i=%d (expected near-zero crossing): %f", i, out[i])
		}
	}
}

func TestResampleAlready16k(t *testing.T) {
	in := []float32{1, 2, 3, 4}
	out, err := ResampleTo16kHz(in, 16000)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(in) {
		t.Fatalf("len = %d, want %d", len(out), len(in))
	}
	for i := range in {
		if out[i] != in[i] {
			t.Errorf("out[%d] = %f, want %f", i, out[i], in[i])
		}
	}
}

func TestTrimSilence(t *testing.T) {
	in := []float32{
		0.001, 0.0005, 0.0, -0.001, 0.001, // trimmed from head
		0.5, -0.5, 0.7,
		0.0, 0.0005, 0.001, // trimmed from tail
	}
	got := TrimSilence(in, 0.002)
	want := []float32{0.5, -0.5, 0.7}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("got[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestPrepareForEngineSilent(t *testing.T) {
	silent := make([]float32, 16000)
	_, err := PrepareForEngine(silent, 16000, 1)
	if err == nil {
		t.Error("expected error for silent audio")
	}
}

func TestPrepareForEngineHappyPath(t *testing.T) {
	// 1 second of audio at 48 kHz stereo (interleaved).
	const sr, ch = 48000, 2
	in := make([]float32, sr) // interleaved frames × 2 channels
	for i := 0; i < len(in); i += ch {
		v := float32(math.Sin(2 * math.Pi * float64(i/ch) / 200))
		in[i] = v
		if ch == 2 {
			in[i+1] = v
		}
	}
	out, err := PrepareForEngine(in, sr, ch)
	if err != nil {
		t.Fatal(err)
	}
	// After mono downmix we have sr/2 frames; after resampling at 3:1
	// we expect roughly sr/6 ≈ 8000 samples at 16 kHz.
	if len(out) == 0 {
		t.Fatal("expected non-empty output")
	}
	if len(out) < 7000 || len(out) > 9000 {
		t.Errorf("resampled length %d outside expected ~8000 range", len(out))
	}
}

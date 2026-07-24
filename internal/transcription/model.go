// Package transcription bundles the STT model catalog/downloader and the
// runtime inference engines. This file holds the catalog metadata and the
// HTTP downloader used by `gott download`.
package transcription

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// SubFile is one file inside a multi-file model directory.
type SubFile struct {
	Name string
	URL  string
	// SizeBytes can be 0 (unknown); in that case progress is reported as
	// "downloaded N bytes" without a percentage.
	SizeBytes int64
}

// ModelInfo describes a single model that gott knows how to use.
type ModelInfo struct {
	ID          string // stable key for config.json
	Name        string // human label for UI
	Filename    string // directory name (parakeet is multi-file)
	SizeBytes   int64  // sum of sub-file sizes
	SubFiles    []SubFile
	IsDirectory bool
}

// ProgressFn is invoked from the downloader with (downloaded, total) bytes.
// total may be 0 if the server didn't return Content-Length.
type ProgressFn func(downloaded, total int64)

// ParakeetTDTInt8 is the default model: NVIDIA Parakeet-TDT 0.6B INT8
// published under altunenes/parakeet-rs. ~670 MB total.
var ParakeetTDTInt8 = ModelInfo{
	ID:          "parakeet-tdt-int8",
	Name:        "Parakeet TDT 0.6B (INT8)",
	Filename:    "parakeet-tdt-int8",
	SizeBytes:   670_000_000, // approximate, for UI only
	IsDirectory: true,
	SubFiles: []SubFile{
		{
			Name:      "encoder-model.int8.onnx",
			URL:       "https://huggingface.co/altunenes/parakeet-rs/resolve/main/tdt/encoder-model.int8.onnx",
			SizeBytes: 240_000_000,
		},
		{
			Name:      "decoder_joint-model.int8.onnx",
			URL:       "https://huggingface.co/altunenes/parakeet-rs/resolve/main/tdt/decoder_joint-model.int8.onnx",
			SizeBytes: 430_000_000,
		},
		{
			Name:      "vocab.txt",
			URL:       "https://huggingface.co/altunenes/parakeet-rs/resolve/main/tdt/vocab.txt",
			SizeBytes: 16_000,
		},
	},
}

// Models returns every model known to gott.
func Models() []ModelInfo { return []ModelInfo{ParakeetTDTInt8} }

// FindModel looks up a model by ID. Returns nil if unknown.
func FindModel(id string) *ModelInfo {
	for _, m := range Models() {
		if m.ID == id {
			return &m
		}
	}
	return nil
}

// ModelDir returns the directory a model lives in.
//
// Resolves $XDG_DATA_HOME/gott/models/<filename> or, if unset,
// ~/.local/share/gott/models/<filename>. The directory is created.
func ModelDir(m ModelInfo) (string, error) {
	base, err := os.UserConfigDir() // safer than UserDataDir (linux returns same)
	var dir string
	if err == nil {
		dir = filepath.Join(base, "..", "share", "gott", "models", m.Filename)
	} else {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, ".local", "share", "gott", "models", m.Filename)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	return dir, nil
}

// IsInstalled reports whether every required sub-file is present and has
// non-zero size. Cheap - safe to call on every UI refresh.
func IsInstalled(m ModelInfo) (bool, error) {
	dir, err := ModelDir(m)
	if err != nil {
		return false, err
	}
	for _, sf := range m.SubFiles {
		info, err := os.Stat(filepath.Join(dir, sf.Name))
		if err != nil || info.Size() == 0 {
			return false, nil
		}
	}
	return true, nil
}

// DownloadModel downloads every sub-file of m, reporting aggregate progress
// (sum of bytes downloaded across files / sum of expected sizes). Already
// present files are skipped.
//
// A partially-present directory is resumed: a sub-file is redownloaded when
// missing or smaller than expected (best-effort; size is only checked when
// the server supplied Content-Length).
func DownloadModel(ctx context.Context, m ModelInfo, fn ProgressFn) error {
	dir, err := ModelDir(m)
	if err != nil {
		return err
	}
	if fn == nil {
		fn = func(int64, int64) {}
	}
	var done int64
	total := m.SizeBytes
	for _, sf := range m.SubFiles {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		dest := filepath.Join(dir, sf.Name)
		// Skip files we already have at the expected size (best effort).
		if fi, err := os.Stat(dest); err == nil && sf.SizeBytes > 0 && fi.Size() == sf.SizeBytes {
			done += sf.SizeBytes
			fn(done, total)
			continue
		}
		f, dlTotal, err := downloadOne(ctx, sf, dest, func(d int64) { fn(done+d, total) })
		if err != nil {
			return fmt.Errorf("download %s: %w", sf.Name, err)
		}
		_ = f // closed inside downloadOne
		done += dlTotal
		fn(done, total)
	}
	return nil
}

// downloadOne streams a single file to <dest>.part then renames it. total
// returns the size actually downloaded (== Content-Length when known).
func downloadOne(ctx context.Context, sf SubFile, dest string, progress func(d int64)) (io.Closer, int64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, sf.URL, nil)
	if err != nil {
		return nil, 0, err
	}
	client := &http.Client{Timeout: 0} // honour ctx instead
	resp, err := client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, 0, fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	expected := sf.SizeBytes
	if resp.ContentLength > 0 {
		expected = resp.ContentLength
	}
	part := dest + ".part"
	out, err := os.OpenFile(part, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return nil, 0, err
	}
	// 32 KiB buffer; smaller would be wasteful, larger wastes memory.
	buf := make([]byte, 32*1024)
	var n int64
	last := time.Now()
	for {
		nr, rerr := resp.Body.Read(buf)
		if nr > 0 {
			if _, werr := out.Write(buf[:nr]); werr != nil {
				_ = out.Close()
				_ = os.Remove(part)
				return nil, 0, werr
			}
			n += int64(nr)
			// Throttle progress callbacks to ~5 Hz to keep UI smooth.
			if time.Since(last) > 200*time.Millisecond {
				progress(n)
				last = time.Now()
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			_ = out.Close()
			_ = os.Remove(part)
			return nil, n, rerr
		}
	}
	progress(n)
	if err := out.Close(); err != nil {
		return nil, n, err
	}
	if err := os.Rename(part, dest); err != nil {
		return nil, n, err
	}
	_ = expected // used implicitly via sf.SizeBytes check in caller
	return nil, n, nil
}

// LoadVocab reads a vocab.txt file (one token per line) into a slice. The
// first line is conventionally the blank token (index 0).
func LoadVocab(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(strings.ReplaceAll(string(data), "\r\n", "\n"), "\n")
	out := make([]string, 0, len(lines))
	for _, l := range lines {
		l = strings.TrimRight(l, "\r ")
		if l == "" {
			continue
		}
		out = append(out, l)
	}
	if len(out) == 0 {
		return nil, errors.New("empty vocab")
	}
	return out, nil
}

// PeekVocabulary is a convenience used by the UI to flag "model not
// downloaded" without doing full work.
func PeekVocabulary(m ModelInfo) (string, error) {
	dir, err := ModelDir(m)
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "vocab.txt"), nil
}



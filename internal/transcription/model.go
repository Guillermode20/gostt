// Package transcription bundles the STT model catalog/downloader and the
// runtime inference engines. This file holds the catalog metadata and the
// HTTP downloader used by `gostt download`.
package transcription

import (
	"bufio"
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

// ModelInfo describes a single model that gostt knows how to use.
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

// Models returns every model known to gostt.
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
// Resolves $XDG_DATA_HOME/gostt/models/<filename> or, if unset,
// ~/.local/share/gostt/models/<filename>. The directory is created.
func ModelDir(m ModelInfo) (string, error) {
	base, err := os.UserConfigDir() // safer than UserDataDir (linux returns same)
	var dir string
	if err == nil {
	dir = filepath.Join(base, "..", "share", "gostt", "models", m.Filename)
	} else {
		home, _ := os.UserHomeDir()
	dir = filepath.Join(home, ".local", "share", "gostt", "models", m.Filename)
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
//
// Improvements over naive io.Copy:
//   - 1 MiB read buffer (vs 32 KiB) — cuts syscall count ~32× for large files.
//   - bufio.Writer — batches small writes into larger disk flushes.
//   - Retry with exponential backoff on transient read errors.
//   - User-Agent header — some CDNs throttle bare requests.
//   - HTTP Range resume — picks up from where .part left off when possible.
func downloadOne(ctx context.Context, sf SubFile, dest string, progress func(d int64)) (io.Closer, int64, error) {
	const (
		readBufSize = 1 << 20 // 1 MiB
		writeBufSize = 1 << 20 // 1 MiB
		maxRetries   = 5
		retryBase    = 2 * time.Second
	)

	part := dest + ".part"

	// Check if we have a partial download to resume from.
	var resumeOffset int64
	if fi, err := os.Stat(part); err == nil && fi.Size() > 0 {
		resumeOffset = fi.Size()
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, sf.URL, nil)
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("User-Agent", "gostt/1.0 (https://github.com/Guillermode20/gostt)")
	if resumeOffset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", resumeOffset))
	}

	client := &http.Client{Timeout: 0} // honour ctx instead
	resp, err := client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()

	// Accept 200 (full) or 206 (partial) responses.
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		return nil, 0, fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	// If server doesn't support resume (returned 200 instead of 206), reset offset.
	if resp.StatusCode == http.StatusOK {
		resumeOffset = 0
	}


	// Open for append if resuming, truncate if fresh.
	flags := os.O_CREATE | os.O_WRONLY
	if resumeOffset == 0 {
		flags |= os.O_TRUNC
	} else {
		flags |= os.O_APPEND
	}
	out, err := os.OpenFile(part, flags, 0o644)
	if err != nil {
		return nil, 0, err
	}

	var totalRead int64 = resumeOffset
	last := time.Now()

	// Wrapped in retry loop for transient network errors.
	for attempt := 0; ; attempt++ {
		buf := make([]byte, readBufSize)
		bw := bufio.NewWriterSize(out, writeBufSize)

		for {
			nr, rerr := resp.Body.Read(buf)
			if nr > 0 {
				if _, werr := bw.Write(buf[:nr]); werr != nil {
					_ = bw.Flush()
					_ = out.Close()
					_ = os.Remove(part)
					return nil, 0, werr
				}
				totalRead += int64(nr)
				if time.Since(last) > 200*time.Millisecond {
					progress(totalRead)
					last = time.Now()
				}
			}
			if rerr == io.EOF {
				break
			}
			if rerr != nil {
				_ = bw.Flush()
				// Retry on transient errors (connection reset, timeout).
				if attempt < maxRetries && isRetryable(rerr) {
					_ = resp.Body.Close()
					wait := retryBase * time.Duration(1<<uint(attempt)) // exponential backoff
					select {
					case <-ctx.Done():
						_ = out.Close()
						return nil, totalRead, ctx.Err()
					case <-time.After(wait):
					}
					// Re-issue request with updated Range header.
					req2, err := http.NewRequestWithContext(ctx, http.MethodGet, sf.URL, nil)
					if err != nil {
						_ = out.Close()
						return nil, totalRead, err
					}
					req2.Header.Set("User-Agent", "gostt/1.0 (https://github.com/Guillermode20/gostt)")
					req2.Header.Set("Range", fmt.Sprintf("bytes=%d-", totalRead))
					resp2, err := client.Do(req2)
					if err != nil {
						_ = out.Close()
						return nil, totalRead, err
					}
					resp = resp2
					continue
				}
				_ = out.Close()
				_ = os.Remove(part)
				return nil, totalRead, rerr
			}
		}
		_ = bw.Flush()
		break // success
	}

	progress(totalRead)
	if err := out.Close(); err != nil {
		return nil, totalRead, err
	}
	if err := os.Rename(part, dest); err != nil {
		return nil, totalRead, err
	}
	return nil, totalRead, nil
}

// isRetryable reports whether err is likely transient and worth retrying.
func isRetryable(err error) bool {
	if errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	var netErr interface{ Timeout() bool }
	if errors.As(err, &netErr) && netErr.Timeout() {
		return true
	}
	// "connection reset by peer" and similar
	s := err.Error()
	return strings.Contains(s, "connection reset") ||
		strings.Contains(s, "broken pipe") ||
		strings.Contains(s, "unexpected EOF")
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



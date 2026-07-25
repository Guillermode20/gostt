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
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
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
// hfToken returns the HuggingFace API token from the environment.
// Authenticated requests get significantly higher rate limits.
func hfToken() string {
	for _, k := range []string{"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"} {
		if v := os.Getenv(k); v != "" {
			return v
		}
	}
	return ""
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
	// XDG_DATA_HOME defaults to ~/.local/share.
	share := os.Getenv("XDG_DATA_HOME")
	if share == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("cannot find home directory: %w", err)
		}
		share = filepath.Join(home, ".local", "share")
	}
	dir := filepath.Join(share, "gostt", "models", m.Filename)
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
		// Skip files we already have (non-zero size).
		if fi, err := os.Stat(dest); err == nil && fi.Size() > 0 {
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
// For files > 100 MB and servers supporting Range requests, the download
// is split across multiple parallel connections to bypass per-connection
// CDN throttling. An HF_TOKEN environment variable enables authenticated
// HuggingFace requests with higher rate limits.
func downloadOne(ctx context.Context, sf SubFile, dest string, progress func(d int64)) (io.Closer, int64, error) {
	const (
		readBufSize     = 1 << 20 // 1 MiB
		writeBufSize    = 1 << 20 // 1 MiB
		maxRetries      = 5
		retryBase       = 2 * time.Second
		parallelMinSize = 100 << 20 // 100 MB — threshold for parallel download
	)


	// Probe the server: can we resume, and does it support Range?
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, sf.URL, nil)
	if err != nil {
		return nil, 0, err
	}
	setHeaders(req)
	client := &http.Client{Timeout: 0}
	resp, err := client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	resp.Body.Close()

	totalSize := sf.SizeBytes
	if resp.ContentLength > 0 {
		totalSize = resp.ContentLength
	}
	supportsRange := resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusPartialContent
	acceptRanges := strings.EqualFold(resp.Header.Get("Accept-Ranges"), "bytes")

	// Decide: parallel or single-stream?
	if totalSize >= parallelMinSize && (acceptRanges || supportsRange) {
		return downloadParallel(ctx, sf, dest, totalSize, progress)
	}
	return downloadSingle(ctx, sf, dest, progress)
}

// downloadParallel splits a file into chunks and downloads them concurrently.
// Each chunk is written to a separate temp file, then assembled into the
// final .part file. This bypasses per-connection throttling on CDNs.
func downloadParallel(ctx context.Context, sf SubFile, dest string, totalSize int64, progress func(d int64)) (io.Closer, int64, error) {
	const (
		readBufSize = 1 << 20
		chunkSize   = 50 << 20 // 50 MB
		maxParallel = 4
	)
	part := dest + ".part"

	// Check for partial resume — how much do we already have?
	var done int64
	if fi, err := os.Stat(part); err == nil {
		done = fi.Size()
	}
	if done >= totalSize {
		// Already complete (shouldn't happen, but be safe).
		if err := os.Rename(part, dest); err != nil {
			return nil, 0, err
		}
		return nil, done, nil
	}

	workers := runtime.NumCPU()
	if workers > maxParallel {
		workers = maxParallel
	}
	if workers < 1 {
		workers = 1
	}

	// Calculate chunk boundaries.
	type chunk struct {
		start, end int64
		index      int
	}
	var chunks []chunk
	for off := done; off < totalSize; {
		end := off + chunkSize
		if end > totalSize {
			end = totalSize
		}
		chunks = append(chunks, chunk{start: off, end: end, index: len(chunks)})
		off = end
	}

	var downloaded atomic.Int64
	downloaded.Store(done)

	// Semaphore to limit concurrency.
	sem := make(chan struct{}, workers)
	var wg sync.WaitGroup
	errs := make([]error, len(chunks))

	for i, c := range chunks {
		wg.Add(1)
		go func(idx int, c chunk) {
			defer wg.Done()
			sem <- struct{}{}        // acquire
			defer func() { <-sem }() // release
			errs[idx] = downloadChunk(ctx, sf.URL, part, c.start, c.end, c.index, &downloaded, progress)
		}(i, c)
	}
	wg.Wait()

	// Check for errors.
	for _, e := range errs {
		if e != nil {
			_ = os.Remove(part)
			return nil, downloaded.Load(), e
		}
	}

	progress(totalSize)
	if err := assembleChunks(part, len(chunks)); err != nil {
		_ = os.Remove(part)
		return nil, downloaded.Load(), err
	}
	if err := os.Rename(part, dest); err != nil {
		return nil, totalSize, err
	}
	return nil, totalSize, nil
}

// downloadChunk downloads a byte range into a temp file, then splices it
// into the main .part file via a write lock.
func downloadChunk(ctx context.Context, url, part string, start, end int64, index int, downloaded *atomic.Int64, progress func(d int64)) error {
	const (
		readBufSize = 1 << 20
		maxRetries  = 3
		retryBase   = 2 * time.Second
	)

	chunkPath := fmt.Sprintf("%s.chunk%d", part, index)
	// If chunk file already exists with correct size, skip.
	if fi, err := os.Stat(chunkPath); err == nil && fi.Size() == (end-start) {
		downloaded.Add(end - start)
		progress(downloaded.Load())
		return nil
	}

	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			wait := retryBase * time.Duration(1<<uint(attempt-1))
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(wait):
			}
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			lastErr = err
			continue
		}
		setHeaders(req)
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", start, end-1))

		client := &http.Client{Timeout: 0}
		resp, err := client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
			resp.Body.Close()
			lastErr = fmt.Errorf("HTTP %d for range %d-%d", resp.StatusCode, start, end-1)
			continue
		}

		f, err := os.OpenFile(chunkPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
		if err != nil {
			resp.Body.Close()
			lastErr = err
			continue
		}

		buf := make([]byte, readBufSize)
		for {
			nr, rerr := resp.Body.Read(buf)
			if nr > 0 {
				if _, werr := f.Write(buf[:nr]); werr != nil {
					f.Close()
					resp.Body.Close()
					lastErr = werr
					break
				}
				downloaded.Add(int64(nr))
				progress(downloaded.Load())
			}
			if rerr == io.EOF {
				break
			}
			if rerr != nil {
				lastErr = rerr
				break
			}
		}
		f.Close()
		resp.Body.Close()
		if lastErr == nil {
			return nil // success
		}
	}
	return fmt.Errorf("chunk %d (%d-%d) failed after retries: %w", index, start, end-1, lastErr)
}

// assembleChunks merges .chunkN files into the .part file, then deletes chunks.
// Must be called after all chunks succeed.
func assembleChunks(part string, numChunks int) error {
	f, err := os.OpenFile(part, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	buf := make([]byte, 1<<20)
	for i := 0; i < numChunks; i++ {
		chunkPath := fmt.Sprintf("%s.chunk%d", part, i)
		cf, err := os.Open(chunkPath)
		if err != nil {
			return fmt.Errorf("open chunk %d: %w", i, err)
		}
		for {
			n, rerr := cf.Read(buf)
			if n > 0 {
				if _, werr := f.Write(buf[:n]); werr != nil {
					cf.Close()
					return werr
				}
			}
			if rerr == io.EOF {
				break
			}
			if rerr != nil {
				cf.Close()
				return rerr
			}
		}
		cf.Close()
		os.Remove(chunkPath)
	}
	return nil
}

// downloadSingle is the original single-stream download with retry/resume.
func downloadSingle(ctx context.Context, sf SubFile, dest string, progress func(d int64)) (io.Closer, int64, error) {
	const (
		readBufSize = 1 << 20
		writeBufSize = 1 << 20
		maxRetries  = 5
		retryBase   = 2 * time.Second
	)

	part := dest + ".part"
	var resumeOffset int64
	if fi, err := os.Stat(part); err == nil && fi.Size() > 0 {
		resumeOffset = fi.Size()
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, sf.URL, nil)
	if err != nil {
		return nil, 0, err
	}
	setHeaders(req)
	if resumeOffset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", resumeOffset))
	}

	client := &http.Client{Timeout: 0}
	resp, err := client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		return nil, 0, fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	if resp.StatusCode == http.StatusOK {
		resumeOffset = 0
	}

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
				if attempt < maxRetries && isRetryable(rerr) {
					_ = resp.Body.Close()
					wait := retryBase * time.Duration(1<<uint(attempt))
					select {
					case <-ctx.Done():
						_ = out.Close()
						return nil, totalRead, ctx.Err()
					case <-time.After(wait):
					}
					req2, err := http.NewRequestWithContext(ctx, http.MethodGet, sf.URL, nil)
					if err != nil {
						_ = out.Close()
						return nil, totalRead, err
					}
					setHeaders(req2)
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
		break
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

// setHeaders adds User-Agent and optional HF_TOKEN auth to a request.
func setHeaders(req *http.Request) {
	req.Header.Set("User-Agent", "gostt/1.0 (https://github.com/Guillermode20/gostt)")
	if token := hfToken(); token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
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



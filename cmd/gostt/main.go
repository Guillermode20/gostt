// Command gostt is the CLI / GUI entrypoint for the offline speech-to-text
// dictation app.
//
// Usage:
//
//	gostt run                   launch GUI + system tray
//	gostt list                  list available microphones
//	gostt download              download the default model (~670 MB)
//	gostt record [seconds]      record N seconds and print transcription
//	gostt --help                this help
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"
	"github.com/Guillermode20/gostt/internal/app"
	"github.com/Guillermode20/gostt/internal/audio"
	"github.com/Guillermode20/gostt/internal/settings"
	"github.com/Guillermode20/gostt/internal/transcription"
	"github.com/Guillermode20/gostt/internal/ui"
)

const usage = `gostt — offline speech-to-text for Linux

USAGE:
  gostt                       launch GUI + system tray
  gostt list                  enumerate microphones
  gostt download              fetch the default Parakeet-TDT model
  gostt record [SECONDS]      record N seconds (default 5) and transcribe
  gostt --help, -h            print this message

ENVIRONMENT:
  GOSTT_ORT_LIB               path to libonnxruntime.so (default: libonnxruntime.so)
  GOSTT_TDT_*                 override ONNX tensor names for non-standard exports
  XDG_SESSION_TYPE           detected automatically; controls input backend choice
`

func init() { loadDotenv() }

// loadDotenv reads .env from the executable's directory (or CWD) and
// sets env vars without overwriting existing ones.
func loadDotenv() {
	// Try executable dir first, then CWD.
	dir := "."
	if ex, err := os.Executable(); err == nil {
		dir = filepath.Dir(ex)
	}
	f, err := os.Open(filepath.Join(dir, ".env"))
	if err != nil {
		// Try CWD as fallback.
		if f2, err2 := os.Open(".env"); err2 == nil {
			f = f2
		} else {
			return
		}
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		k, v, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		k = strings.TrimSpace(k)
		v = strings.TrimSpace(v)
		// Strip surrounding quotes.
		if len(v) >= 2 && (v[0] == '"' && v[len(v)-1] == '"' || v[0] == '\'' && v[len(v)-1] == '\'') {
			v = v[1 : len(v)-1]
		}
		// Don't overwrite existing env vars.
		if _, exists := os.LookupEnv(k); !exists {
			os.Setenv(k, v)
		}
	}
}
func main() {
	// Use stdlib flag for --help/-h so we get a uniform UX.
	if len(os.Args) >= 2 && (os.Args[1] == "-h" || os.Args[1] == "--help") {
		fmt.Print(usage)
		return
	}

	args := os.Args[1:]
	if len(args) == 0 {
		if err := runGUI(); err != nil {
			fail(err)
		}
		return
	}

	switch args[0] {
	case "run":
		if err := runGUI(); err != nil {
			fail(err)
		}
	case "list":
		runList()
	case "download":
		runDownload()
	case "record":
		var seconds float64
		if len(args) >= 2 {
			if _, err := fmt.Sscanf(args[1], "%f", &seconds); err != nil {
				fail(fmt.Errorf("invalid seconds: %s", args[1]))
			}
		}
		if seconds <= 0 {
			seconds = 5
		}
		if err := runRecord(seconds); err != nil {
			fail(err)
		}
	default:
		fmt.Print(usage)
		os.Exit(2)
	}
}

func fail(err error) {
	fmt.Fprintf(os.Stderr, "gostt: %v\n", err)
	os.Exit(1)
}

// runGUI boots the orchestrator + Fyne window + tray in one go.
func runGUI() error {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sig
		fmt.Fprintln(os.Stderr, "\nshutting down…")
		cancel()
	}()

	a, err := app.New(app.Options{
		AppID: "io.github.guillermode20.gostt",
		Title: "gostt STT",
	})
	if err != nil {
		return err
	}
	if err := a.Start(ctx, nil); err != nil {
		return err
	}

	state := ui.NewState()
	cb := &ui.Callbacks{
		OnStart: func(device string) {
			deviceCopy := device
			a.Send(app.StartRecording{Device: &deviceCopy})
		},
		OnStop:     func() { a.Send(app.StopRecording{}) },
		OnDownload: func() { a.Send(app.DownloadModel{}) },
		OnClear:    func() { _ = state.Transcription.Set("") },
		OnSaveSettings: func(s settings.Settings) {
			// Re-bind the XDG GlobalShortcuts Portal session to the new trigger.
			a.Send(app.ReflectHotkey{Trigger: s.HoldToTalkKey})
		},
		OnUnselectMic: func() { a.SetMic("") },
		OnQuit:        func() { a.Send(app.Shutdown{}) },
	}
	a.SetTrayCallbacks(
		func() { a.Send(app.StartRecording{Device: nil}) }, // hotkey re-press; alternatively ToggleWindow()
		func() { a.Send(app.Shutdown{}) },
	)
	_ = cb
	if err := ui.RunWindow(a, state, cb, "gostt STT"); err != nil {
		return err
	}
	select {
	case <-a.ShutdownChannel():
	default:
		a.RequestShutdown()
	}
	_ = a.Shutdown(2 * time.Second)
	return nil
}

// runList prints a table of detected input devices.
func runList() {
	mics, err := audio.ListMicrophones()
	if err != nil {
		fail(err)
	}
	if len(mics) == 0 {
		fmt.Println("no microphones found.")
		return
	}
	fmt.Printf("%-4s %-50s %-10s %-10s\n", "id", "name", "rate", "channels")
	for i, m := range mics {
		mark := " "
		if m.IsDefault {
			mark = "*"
		}
		fmt.Printf("[%-2s] %-50s %-10d %-10d\n", mark, m.Name, m.SampleRate, m.Channels)
		_ = i
	}
	fmt.Println("\n(*) system default device")
}

// runDownload pulls the default model into the XDG data dir.
func runDownload() {
	ctx := context.Background()
	m := transcription.ParakeetTDTInt8
	ok, err := transcription.IsInstalled(m)
	if err != nil {
		fail(err)
	}
	if ok {
		fmt.Println("model already installed at:")
		dir, _ := transcription.ModelDir(m)
		fmt.Println("  ", dir)
		return
	}
	fmt.Printf("downloading %s (~%d MB) ...\n", m.Name, m.SizeBytes/1024/1024)
	t0 := time.Now()
	err = transcription.DownloadModel(ctx, m, func(d, total int64) {
		if total > 0 {
			fmt.Printf("\r %5.1f %%  %d / %d MB      ", float64(d)/float64(total)*100, d/1024/1024, total/1024/1024)
		}
	})
	fmt.Println()
	if err != nil {
		fail(err)
	}
	dir, _ := transcription.ModelDir(m)
	fmt.Printf("done in %s -> %s\n", time.Since(t0).Round(time.Second), dir)
}

// runRecord captures audio for `seconds`, then transcribes.
func runRecord(seconds float64) error {
	fmt.Printf("recording %g seconds on default microphone...\n", seconds)
	pcm, rate, err := audio.RecordSeconds(context.Background(), "", seconds)
	if err != nil {
		return err
	}
	fmt.Printf("captured %d samples @ %d Hz\n", len(pcm), rate)
	prepared, err := audio.PrepareForEngine(pcm, int(rate), 1)
	if err != nil {
		return err
	}
	fmt.Printf("prepared %d samples @ 16 kHz mono\n", len(prepared))

	m := transcription.ParakeetTDTInt8
	ok, err := transcription.IsInstalled(m)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("model not installed; run `gostt download` first")
	}
	eng, err := transcription.NewEngine(m, transcription.EngineConfig{
		Threads: min(runtime.NumCPU(), 4),
	})
	if err != nil {
		return err
	}
	defer eng.Close()

	t0 := time.Now()
	res, err := eng.Transcribe(prepared)
	if err != nil {
		return err
	}
	dur := time.Since(t0)
	fmt.Println()
	fmt.Println("---- transcript ----")
	fmt.Println(res.Text)
	fmt.Println("----")
	fmt.Printf("tokens : %d\n", res.TokensEmitted)
	fmt.Printf("time   : %s\n", dur.Round(10*time.Millisecond))
	return nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// flag is imported to satisfy the "additional flag arguments" UX if we
// ever expand the record subcommand with flags.
var _ = flag.NewFlagSet

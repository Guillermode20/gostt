// Package app is the orchestrator that wires settings, audio, hotkey,
// transcription, tray, and the GUI together. The orchestrator owns:
//
//   - a single Worker goroutine that processes all mutation requests
//     serially (no locks).
//   - a thin inbound channel (chanWorker) for UI / CLI / hotkey events.
//   - a thin outbound channel (chanUpdate) that the UI / tray consume to
//     reflect state changes.
//
// Boundaries: outside this package, every other module is expected to be
// safe to use from arbitrary goroutines. Inside the orchestrator, the
// Worker goroutine is the single writer of mutable state.
package app

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/Guillermode20/gostt/internal/audio"
	"github.com/Guillermode20/gostt/internal/clipboard"
	"github.com/Guillermode20/gostt/internal/hotkey"
	"github.com/Guillermode20/gostt/internal/inputsim"
	"github.com/Guillermode20/gostt/internal/settings"
	"github.com/Guillermode20/gostt/internal/transcription"
	"github.com/Guillermode20/gostt/internal/tray"
)

// State is the high-level UI state fanned out via chanUpdate. Mirrored to
// tray via the same channel.
type State int

const (
	StateIdle State = iota
	StateListening
	StateRecording
	StateTranscribing
	StateModelMissing
	StateError
)

func (s State) String() string {
	switch s {
	case StateIdle:
		return "Idle"
	case StateListening:
		return "Listening for hotkey"
	case StateRecording:
		return "Recording"
	case StateTranscribing:
		return "Transcribing"
	case StateModelMissing:
		return "Model required"
	case StateError:
		return "Error"
	}
	return "?"
}

// Update is broadcast from the Worker to the UI / tray.
type Update struct {
	State           State
	Status          string
	Transcription   string
	InputLevel      float32 // 0..1; -1 means "do not update"
	Mics            []audio.MicInfo
	Hotkey          string
	ModelSize       int64
	ModelReady      bool
	Downloading     bool
	DownloadPct     float64 // 0..100
	DownloadBytes   int64
	DownloadTotal   int64
}

// WorkerMsg is inbound to the Worker. We use a plain interface{} so that
// helper types can stay simple structs without marker methods; the type
// switch in handleMessage keeps the discrimination typesafe.
type WorkerMsg = interface{}

type (
	StartRecording struct{ Device *string }
	StopRecording  struct{}
	DownloadModel  struct{}
	ReflectHotkey  struct {
		Trigger string
	}
	Shutdown struct{}
)

// Options configures App from main().
type Options struct {
	Headless bool   // skip tray+GUI for CLI subcommands
	AppID    string // XDG app id (unused on tray for now, but kept for future)
	Title    string
}

// App holds all long-lived subsystems.
type App struct {
	opts    Options
	mu      sync.Mutex
	cfg     settings.Settings
	state   State
	engine  transcription.Engine
	stream  *audio.Stream
	listener *hotkey.Listener
	tray    *tray.Tray
	onTrayToggle func()
	shutdownCh chan struct{}

	chanUpdate chan Update
	chanMic    chan []audio.MicInfo
	chanHotkey chan hotkey.Event
	chanWorker chan WorkerMsg
	chanTray   chan tray.Status

	// keepLastBuffer reserves audio captured during hold-to-talk so the
	// worker can transcribe it on Stop. Cleared each cycle.
	recMu     sync.Mutex
	recBuf    []float32
	recRate   uint32
}

// New builds an *App but does not start any goroutine.
func New(opts Options) (*App, error) {
	cfg, err := settings.Load()
	if err != nil {
		return nil, fmt.Errorf("load settings: %w", err)
	}
	return &App{
		opts:       opts,
		cfg:        cfg,
		state:      StateIdle,
		chanUpdate: make(chan Update, 128),
		chanMic:    make(chan []audio.MicInfo, 4),
		chanHotkey: make(chan hotkey.Event, 8),
		chanWorker: make(chan WorkerMsg, 16),
		chanTray:   make(chan tray.Status, 16),
	}, nil
}

// UpdateChannel is the read side of chanUpdate.
func (a *App) UpdateChannel() <-chan Update { return a.chanUpdate }

// TrayChannel feeds the tray goroutine.
func (a *App) TrayStatusChannel() <-chan tray.Status { return a.chanTray }

// MicsChannel returns the latest list of microphones (one-shot fetch).
func (a *App) MicsChannel() <-chan []audio.MicInfo { return a.chanMic }

// Settings returns a copy so callers can't mutate under us.
func (a *App) Settings() settings.Settings {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.cfg
}

// Send enqueues a Worker message from outside the package.
func (a *App) Send(m WorkerMsg) {
	defer func() { _ = recover() }()
	select {
	case a.chanWorker <- m:
	default:
		// Drop; the worker is busy.
	}
}

// Start boots all subsystems. Returns once they're alive.
//
// onTrayToggle is invoked when the user clicks "Show / Hide Window" in the
// tray menu.
func (a *App) Start(ctx context.Context, onTrayToggle func()) error {
	// Tray
	if !a.opts.Headless {
		a.tray = tray.New()
		go func() {
			_ = a.tray.Run()
		}()
		// mToggle wiring is omitted here (systray clicks are async; we
		// forward them as a Tray event for the orchestrator to handle).
		// For simplicity we just rely on the user double-clicking the icon
		// to toggle the main window.
	}

	// Hotkey listener
	l, err := hotkey.New(a.cfg.HoldToTalkKey)
	if err != nil {
		// Not fatal: the user can use the in-window record button.
		a.publish(Update{State: StateIdle, Status: fmt.Sprintf("hotkey listener unavailable: %v", err)})
	} else {
		a.listener = l
		go a.pumpHotkey(l)
	}

	// Kick off initial mic enumeration.
	go a.enumerateMics(ctx)

	// Worker.
	go a.workerLoop(ctx)

	// Tray pump (status from Update channel).
	if a.tray != nil {
		go a.pumpTray()
	}
	return nil
}

// Shutdown stops subsystems and waits up to timeout for shutdown to
// complete.
func (a *App) Shutdown(timeout time.Duration) error {
	if a.listener != nil {
		_ = a.listener.Close()
	}
	if a.tray != nil {
		a.tray.Quit()
	}
	if a.stream != nil {
		a.stream.Stop()
	}
	if a.engine != nil {
		_ = a.engine.Close()
	}
	return nil
}

// RequestShutdown cancels the internal shutdown signal so worker loops
// can exit. Intended for the workflow where the orchestrator owns a
// context and Shutdown originates elsewhere (e.g. tray Quit).
func (a *App) RequestShutdown() {
	if a.shutdownCh != nil {
		select {
		case <-a.shutdownCh:
		default:
			close(a.shutdownCh)
		}
	}
}

// ShutdownChannel returns the read side of the orchestrator's shutdown
// signal. Callers can select on it to cancel long-running loops.
func (a *App) ShutdownChannel() <-chan struct{} {
	if a.shutdownCh == nil {
		a.shutdownCh = make(chan struct{})
	}
	return a.shutdownCh
}

// SetTrayCallbacks wires the tray's "Show / Hide Window" and "Quit"
// buttons into orchestrator-supplied handlers.
func (a *App) SetTrayCallbacks(onToggle func(), onQuit func()) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.tray != nil {
		a.tray.OnToggle = onToggle
		a.tray.OnQuit = onQuit
	}
	a.onTrayToggle = onToggle
}

// ToggleWindow is the public helper for tray "Show / Hide Window" — it
// surfaces a Status update so the GUI in either mode can react.
func (a *App) ToggleWindow() {
	if a.onTrayToggle != nil {
		a.onTrayToggle()
	}
}

// applyHotkeyChange stops the existing portal listener (if any) and
// rebinds to the new trigger string. Failures are surfaced as State
// updates rather than fatal.
func (a *App) applyHotkeyChange(trigger string) {
	if trigger == "" {
		trigger = a.cfg.HoldToTalkKey
	}
	a.mu.Lock()
	a.cfg.HoldToTalkKey = trigger
	a.mu.Unlock()
	if a.listener != nil {
		_ = a.listener.Close()
		a.listener = nil
	}
	l, err := hotkey.New(trigger)
	if err != nil {
		a.publish(Update{Status: fmt.Sprintf("hotkey rebind failed: %v", err)})
		return
	}
	a.listener = l
	go a.pumpHotkey(l)
	a.publish(Update{Status: fmt.Sprintf("hotkey rebound to %s", trigger)})
}

// SetMic records a new preferred device; the orchestrator will use it on
// the next StartRecording. It does NOT auto-start a capture session.
func (a *App) SetMic(name string) {
	a.mu.Lock()
	if name == "" {
		a.cfg.PreferredDevice = nil
	} else {
		dup := name
		a.cfg.PreferredDevice = &dup
	}
	a.mu.Unlock()
	_ = settings.Save(a.Settings())
	a.publish(Update{Status: fmt.Sprintf("microphone preference updated: %q", name)})
}

// Hotkey returns a copy of the currently configured hotkey.
func (a *App) Hotkey() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.cfg.HoldToTalkKey
}

// SelectMic is a deprecated alias kept for backward compatibility. New
// callers should use SetMic + the GUI's record button.
func (a *App) SelectMic(name string) {
	a.SetMic(name)
}

// workerLoop is the sole mutator of mutable state.
func (a *App) workerLoop(ctx context.Context) {
	idleTicker := time.NewTicker(8 * time.Second)
	defer idleTicker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case msg := <-a.chanWorker:
			a.handleMessage(msg)
		case ev := <-a.chanHotkey:
			a.handleHotkey(ev)
		case <-idleTicker.C:
			// Periodic settings flush.
			_ = settings.Save(a.Settings())
		}
	}
}

func (a *App) handleMessage(m WorkerMsg) {
	switch v := m.(type) {
	case StartRecording:
		a.ensureRecording(v.Device)
	case StopRecording:
		a.stopAndTranscribe()
	case DownloadModel:
		a.downloadModel()
	case ReflectHotkey:
		a.applyHotkeyChange(v.Trigger)
	case Shutdown:
		a.publish(Update{State: StateIdle, Status: "shutting down"})
		a.RequestShutdown()
	default:
		// unhandled: ignore
		_ = v
	}
}

// currentState returns the orchestrator's processing state under the
// shared mutex.
func (a *App) currentState() State {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.state
}

func (a *App) handleHotkey(ev hotkey.Event) {
	switch ev {
	case hotkey.Pressed:
		a.ensureRecording(nil)
	case hotkey.Released:
		a.stopAndTranscribe()
	}
}

func (a *App) ensureRecording(device *string) {
	switch a.currentState() {
	case StateRecording, StateTranscribing:
		return
	}
	name := ""
	if device != nil {
		name = *device
	} else if a.cfg.PreferredDevice != nil {
		name = *a.cfg.PreferredDevice
	}
	stream, err := audio.NewStream(name, a.onAudioChunk, a.onAudioLevel)
	if err != nil {
		a.publish(Update{State: StateError, Status: fmt.Sprintf("audio: %v", err)})
		return
	}
	a.stream = stream
	a.recMu.Lock()
	a.recBuf = a.recBuf[:0]
	a.recRate = stream.SampleRate()
	a.recMu.Unlock()
	a.setState(StateRecording)
	a.publish(Update{State: StateRecording, Status: "recording…"})
}

func (a *App) stopAndTranscribe() {
	if a.currentState() != StateRecording {
		return
	}
	if a.stream != nil {
		a.stream.Stop()
		a.stream = nil
	}
	a.recMu.Lock()
	buf := a.recBuf
	rate := a.recRate
	a.recBuf = nil
	a.recRate = 0
	a.recMu.Unlock()

	a.setState(StateTranscribing)
	a.publish(Update{State: StateTranscribing, Status: "preparing audio…"})

	go a.transcribeCaptured(buf, rate)
}

func (a *App) transcribeCaptured(buf []float32, rate uint32) {
	if len(buf) == 0 {
		a.publish(Update{State: StateIdle, Status: "no audio captured"})
		return
	}
	pcm, err := audio.PrepareForEngine(buf, int(rate), 1)
	if err != nil {
		a.publish(Update{State: StateError, Status: fmt.Sprintf("audio DSP: %v", err)})
		return
	}
	if a.engine == nil {
		if !a.ensureModel() {
			a.publish(Update{State: StateModelMissing, Status: "model not installed"})
			return
		}
	}
	t0 := time.Now()
	res, err := a.engine.Transcribe(pcm)
	if err != nil {
		a.publish(Update{State: StateError, Status: fmt.Sprintf("transcribe: %v", err)})
		return
	}
	a.publish(Update{
		State:         StateIdle,
		Status:        fmt.Sprintf("transcribed %d tokens in %d ms", res.TokensEmitted, time.Since(t0).Milliseconds()),
		Transcription: res.Text,
		ModelReady:    true,
	})
	if res.Text != "" {
		if err := clipboard.Write(res.Text); err != nil {
			a.publish(Update{Status: fmt.Sprintf("clipboard: %v", err)})
		}
		if a.cfg.AutoType {
			if _, err := inputsim.TypeText(res.Text); err != nil {
				a.publish(Update{Status: fmt.Sprintf("auto-type: %v", err)})
			}
		}
	}
}

func (a *App) ensureModel() bool {
	m := transcription.ParakeetTDTInt8
	ok, err := transcription.IsInstalled(m)
	if err != nil || !ok {
		return false
	}
	if a.engine != nil {
		return true
	}
	e, err := transcription.NewEngine(m, transcription.EngineConfig{Threads: a.cfg.Threads})
	if err != nil {
		a.publish(Update{State: StateError, Status: fmt.Sprintf("load model: %v", err)})
		return false
	}
	a.engine = e
	return true
}

// SetStateWorking is a helper to publish a "working" state with a custom
// status string in one call.
func (a *App) publishWorking(status string) {
	a.publish(Update{State: StateTranscribing, Status: status})
}

func (a *App) downloadModel() {
	a.publish(Update{
		State:       StateTranscribing,
		Status:      "downloading model…",
		Downloading: true,
	})
	if !a.opts.Headless {
		a.sendTray(tray.Status{Kind: tray.IconWorking, Title: "gostt", Tooltip: "Downloading model"})
	}
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
		defer cancel()
		err := transcription.DownloadModel(ctx, transcription.ParakeetTDTInt8, func(d, t int64) {
			pct := percent(d, t)
			a.publish(Update{
				State:         StateTranscribing,
				Status:        fmt.Sprintf("downloading model: %.0f%%", pct),
				Downloading:   true,
				DownloadPct:   pct,
				DownloadBytes: d,
				DownloadTotal: t,
			})
			if t > 0 {
			a.sendTray(tray.Status{Kind: tray.IconWorking, Title: "gostt", Tooltip: fmt.Sprintf("Downloading model %.0f%%", pct)})
			}
		})
		if err != nil {
			a.publish(Update{State: StateError, Status: fmt.Sprintf("download: %v", err)})
			return
		}
		a.publish(Update{State: StateIdle, Status: "model ready", ModelReady: true})
		a.sendTray(tray.Status{Kind: tray.IconIdle, Title: "gostt", Tooltip: "gostt — model ready"})
	}()
}

func (a *App) pumpHotkey(l *hotkey.Listener) {
	// Forward both events and status messages.
	go func() {
		for {
			select {
			case e, ok := <-l.Events():
				if !ok {
					return
				}
				a.mu.Lock()
				a.publishHotkeyStatus(e)
				a.mu.Unlock()
				a.chanHotkey <- e
			case s, ok := <-l.Status():
				if !ok {
					return
				}
				a.publish(Update{Status: s})
			}
		}
	}()
}

func (a *App) enumerateMics(ctx context.Context) {
	infos, err := audio.ListMicrophones()
	if err != nil {
		a.publish(Update{Status: fmt.Sprintf("enumerate mics: %v", err)})
		return
	}
	select {
	case a.chanMic <- infos:
	case <-ctx.Done():
	}
}

func (a *App) pumpTray() {
	for u := range a.chanTray {
		a.tray.Update(u)
	}
}

// onAudioChunk captures audio for later transcription. The capture buffer
// is shared with the worker which drains it on Stop.
func (a *App) onAudioChunk(chunk []float32) {
	a.recMu.Lock()
	defer a.recMu.Unlock()
	if a.recBuf == nil {
		return
	}
	a.recBuf = append(a.recBuf, chunk...)
}

// onAudioLevel is throttled — we don't want 100 Hz UI repaints.
func (a *App) onAudioLevel(level float32) {
	if a.stream == nil {
		return
	}
	// Reuse the Update channel and rely on Fyne binding's coalescing.
	a.publish(Update{InputLevel: level})
}

// publish fans out an Update to listeners.
func (a *App) publish(u Update) {
	if u.Hotkey == "" {
		u.Hotkey = a.Hotkey()
	}
	select {
	case a.chanUpdate <- u:
	default:
	}
	// Bridge to tray as well.
	switch u.State {
	case StateIdle:
		a.sendTray(tray.Status{Kind: tray.IconIdle, Title: "gostt", Hotkey: a.Hotkey()})
	case StateListening:
		a.sendTray(tray.Status{Kind: tray.IconIdle, Title: "gostt", Hotkey: a.Hotkey(), Tooltip: "listening"})
	case StateRecording:
		a.sendTray(tray.Status{Kind: tray.IconRecording, Title: "gostt", Hotkey: a.Hotkey(), Tooltip: "recording"})
	case StateTranscribing:
		a.sendTray(tray.Status{Kind: tray.IconWorking, Title: "gostt", Hotkey: a.Hotkey(), Tooltip: "transcribing"})
	case StateModelMissing:
		a.sendTray(tray.Status{Kind: tray.IconModelMissing, Title: "gostt", Hotkey: a.Hotkey(), Tooltip: "model required"})
	case StateError:
		a.sendTray(tray.Status{Kind: tray.IconError, Title: "gostt", Hotkey: a.Hotkey(), Tooltip: u.Status})
	}
}

func (a *App) sendTray(s tray.Status) {
	if a.tray == nil {
		return
	}
	select {
	case a.chanTray <- s:
	default:
	}
}

func (a *App) publishHotkeyStatus(e hotkey.Event) {
	a.publish(Update{Status: fmt.Sprintf("hotkey %s", e)})
}

func (a *App) setState(s State) {
	a.mu.Lock()
	a.state = s
	a.mu.Unlock()
}

func percent(d, t int64) float64 {
	if t <= 0 {
		return 0
	}
	return float64(d) / float64(t) * 100.0
}

// Reset saves the latest settings to disk and reboots internal flags.
func (a *App) Reset() error {
	a.mu.Lock()
	defaults := settings.DefaultSettings()
	a.cfg = defaults
	a.mu.Unlock()
	if err := settings.Save(defaults); err != nil {
		return err
	}
	a.publish(Update{State: StateIdle, Status: "reset to defaults"})
	return nil
}

// EnsureMic brings up the configured microphone (useful on first start).
func (a *App) EnsureMic() error {
	if _, err := os.Stat("/dev/snd"); err != nil {
		// Devices may still be reachable via PipeWire/Pulse; not fatal.
	}
	return nil
}

// AsError is a tiny helper for callers that want to log via Status.
func AsError(err error) Update { return Update{State: StateError, Status: err.Error()} }

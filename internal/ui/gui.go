// Package ui wraps the Fyne v2 GUI around the orchestrator's Update
// channel. Writes back from widgets (e.g., hotkey changes)
// go through the orchestrator's WorkerMsg channel.
package ui

import (
	"context"
	"fmt"
	"strings"
	"time"

	"fyne.io/fyne/v2"
	fyneApp "fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/data/binding"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/widget"

	gottApp "github.com/Guillermode20/gostt/internal/app"
	"github.com/Guillermode20/gostt/internal/audio"
	"github.com/Guillermode20/gostt/internal/clipboard"
	"github.com/Guillermode20/gostt/internal/settings"
)

// Callbacks lets the GUI send messages back to the orchestrator's worker.
type Callbacks struct {
	OnStart        func(deviceName string)
	OnStop         func()
	OnDownload     func()
	OnCopy         func()
	OnClear        func()
	OnSaveSettings func(s settings.Settings)
	OnUnselectMic  func()
	OnQuit         func()
}

// State holds every reactive binding the GUI reads from.
type State struct {
	Status        binding.String
	InputLevel    binding.Float
	Transcription binding.String
	ModelReady    binding.Bool
	Hotkey        binding.String
	Threads       binding.Int
	AutoType      binding.Bool
	SelectedMic   binding.String
	ModelID       binding.String
	IsRecording   binding.Bool
}

func NewState() *State {
	return &State{
		Status:        binding.NewString(),
		InputLevel:    binding.NewFloat(),
		Transcription: binding.NewString(),
		ModelReady:    binding.NewBool(),
		Hotkey:        binding.NewString(),
		Threads:       binding.NewInt(),
		AutoType:      binding.NewBool(),
		SelectedMic:   binding.NewString(),
		ModelID:       binding.NewString(),
		IsRecording:   binding.NewBool(),
	}
}

// Window ties together the Fyne widgets.
type Window struct {
	a       *gottApp.App
	fyneApp fyne.App
	w       fyne.Window

	state *State
	cb    *Callbacks

	recBtn        *widget.Button
	dlBtn         *widget.Button
	copyBtn       *widget.Button
	clearBtn      *widget.Button
	micSelect     *widget.Select
	levelBar      *widget.ProgressBar
	dlBar         *widget.ProgressBar
	dlLabel       *widget.Label
	dlSection     *fyne.Container
	transcript    *widget.Label
	statusLabel   *widget.Label
	hotkeyEntry   *widget.Entry
	threadsEntry  *widget.Entry
	autoTypeCheck *widget.Check
	settingsBox   *fyne.Container

	wantsQuit  bool
	pollerStop chan struct{}
}

// RunWindow blocks until the user quits.
func RunWindow(a *gottApp.App, state *State, cb *Callbacks, title string) error {
	w := &Window{
		a:          a,
		fyneApp:    fyneApp.NewWithID("io.github.guillermode20.gostt"),
		state:      state,
		cb:         cb,
		pollerStop: make(chan struct{}),
	}

	// Apply dark monospace theme.
	w.fyneApp.Settings().SetTheme(NewDarkMonospace())

	w.w = w.fyneApp.NewWindow(title)
	w.w.Resize(fyne.NewSize(560, 480))
	w.w.SetPadded(false)
	w.w.SetCloseIntercept(func() { w.w.Hide() })

	w.buildUI()
	w.attachBindings()

	if cfg, err := settings.Load(); err == nil {
		_ = state.Hotkey.Set(cfg.HoldToTalkKey)
		_ = state.Threads.Set(cfg.Threads)
		_ = state.AutoType.Set(cfg.AutoType)
		_ = state.ModelID.Set(cfg.Model)
		w.autoTypeCheck.SetChecked(cfg.AutoType)
	}

	w.w.Show()

	go w.pollUpdates()
	w.fyneApp.Run()

	close(w.pollerStop)
	return nil
}

// thinSep returns a minimal horizontal separator.
func thinSep() *widget.Separator {
	return widget.NewSeparator()
}

func (w *Window) buildUI() {
	// ── Status line ───────────────────────────────────────
	w.statusLabel = widget.NewLabel("starting…")
	w.statusLabel.Wrapping = fyne.TextWrapWord

	// ── Mic row ───────────────────────────────────────────
	w.micSelect = widget.NewSelect([]string{}, func(name string) {
		if w.cb != nil && w.cb.OnClear != nil {
			w.cb.OnClear()
		}
	})
	w.micSelect.PlaceHolder = "mic"
	refreshBtn := widget.NewButton("↻", func() {
		if w.cb != nil && w.cb.OnClear != nil {
			w.cb.OnClear()
		}
		if w.cb != nil && w.cb.OnUnselectMic != nil {
			w.cb.OnUnselectMic()
		}
	})
	refreshBtn.Importance = widget.LowImportance
	micRow := container.NewBorder(nil, nil, nil, refreshBtn, w.micSelect)

	// ── Level bar ─────────────────────────────────────────
	w.levelBar = widget.NewProgressBar()
	w.levelBar.Min = 0
	w.levelBar.Max = 1

	// ── Record button ─────────────────────────────────────
	w.recBtn = widget.NewButton("● RECORD", func() { w.toggleRecord() })
	w.recBtn.Importance = widget.HighImportance

	// ── Download section ──────────────────────────────────
	w.dlBar = widget.NewProgressBar()
	w.dlBar.Min = 0
	w.dlBar.Max = 1
	w.dlBar.SetValue(0)

	w.dlLabel = widget.NewLabel("")

	w.dlBtn = widget.NewButton("Download Model", func() {
		if w.cb != nil && w.cb.OnDownload != nil {
			w.cb.OnDownload()
		}
	})

	dlInfo := container.NewBorder(nil, nil, nil, w.dlLabel, w.dlBar)
	w.dlSection = container.NewBorder(nil, nil, nil, w.dlBtn, dlInfo)

	// Initially show download button, hide progress bar.
	w.dlBtn.Show()

	// ── Transcript ────────────────────────────────────────
	w.transcript = widget.NewLabel("")
	w.transcript.Wrapping = fyne.TextWrapWord
	w.transcript.TextStyle = fyne.TextStyle{Monospace: true}

	// ── Copy / Clear row ──────────────────────────────────
	w.copyBtn = widget.NewButton("Copy", func() {
		s, _ := w.state.Transcription.Get()
		if s == "" {
			return
		}
		_ = clipboard.Write(s)
		w.setStatus("copied")
	})
	w.clearBtn = widget.NewButton("Clear", func() {
		if w.cb != nil && w.cb.OnClear != nil {
			w.cb.OnClear()
		}
		w.transcript.SetText("")
	})

	// ── Settings (collapsed) ──────────────────────────────
	w.buildSettingsPanel()

	// ── Compose layout ────────────────────────────────────
	topSection := container.NewVBox(
		w.statusLabel,
		micRow,
		w.recBtn,
		w.dlSection,
		container.NewPadded(w.levelBar),
		thinSep(),
	)

	scrollTranscript := container.NewVScroll(w.transcript)

	actionRow := container.NewHBox(w.copyBtn, w.clearBtn)

	content := container.NewBorder(
		topSection,          // top
		container.NewVBox(   // bottom
			thinSep(),
			actionRow,
			w.settingsBox,
		),                  // bottom
		nil, nil,            // left, right
		scrollTranscript,   // center
	)
	w.w.SetContent(content)
}

func (w *Window) buildSettingsPanel() {
	w.hotkeyEntry = widget.NewEntry()
	w.hotkeyEntry.SetText(settings.DefaultHotkey)
	w.hotkeyEntry.OnChanged = func(s string) {
		s = strings.TrimSpace(s)
		if s == "" {
			return
		}
		_ = w.state.Hotkey.Set(s)
	}

	w.threadsEntry = widget.NewEntry()
	w.threadsEntry.SetText("4")
	w.threadsEntry.Validator = func(s string) error {
		if s == "" {
			return nil
		}
		var n int
		_, err := fmt.Sscanf(s, "%d", &n)
		if err != nil || n < 1 || n > 8 {
			return fmt.Errorf("1..8")
		}
		return nil
	}
	w.threadsEntry.OnChanged = func(s string) {
		var n int
		_, _ = fmt.Sscanf(s, "%d", &n)
		_ = w.state.Threads.Set(n)
	}

	w.autoTypeCheck = widget.NewCheck("auto-type", func(on bool) {
		_ = w.state.AutoType.Set(on)
	})

	saveBtn := widget.NewButton("Save", w.saveSettings)
	saveBtn.Importance = widget.LowImportance

	settingsForm := container.NewVBox(
		container.NewHBox(widget.NewLabel("hotkey:"), w.hotkeyEntry),
		container.NewHBox(widget.NewLabel("threads:"), w.threadsEntry),
		w.autoTypeCheck,
		saveBtn,
	)
	settingsItem := widget.NewAccordionItem("settings", settingsForm)
	w.settingsBox = container.NewVBox(widget.NewAccordion(settingsItem))
}

func (w *Window) saveSettings() {
	cfg, _ := settings.Load()
	hk, _ := w.state.Hotkey.Get()
	t, _ := w.state.Threads.Get()
	at, _ := w.state.AutoType.Get()
	if hk != "" {
		cfg.HoldToTalkKey = hk
	}
	if t > 0 {
		cfg.Threads = t
	}
	cfg.AutoType = at
	if err := settings.Save(cfg); err != nil {
		fyne.Do(func() { dialog.ShowError(err, w.w) })
		return
	}
	if w.cb != nil && w.cb.OnSaveSettings != nil {
		w.cb.OnSaveSettings(cfg)
	}
	fyne.Do(func() { w.setStatus("settings saved") })
}

func (w *Window) attachBindings() {
	ch := w.a.UpdateChannel()

	go func() {
		for u := range ch {
			w.apply(u)
		}
	}()

	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		select {
		case mics := <-w.a.MicsChannel():
			w.populateMics(mics)
		case <-ctx.Done():
		}
	}()
}

func (w *Window) apply(u gottApp.Update) {
	if u.Status != "" {
		fyne.Do(func() { w.setStatus(u.Status) })
	}
	if u.Transcription != "" {
		_ = w.state.Transcription.Set(u.Transcription)
		fyne.Do(func() { w.transcript.SetText(u.Transcription) })
	}

	// Download progress.
	if u.Downloading {
		fyne.Do(func() {
			w.dlBtn.Hide()
			w.dlBar.Show()
			w.dlBar.SetValue(u.DownloadPct / 100.0)
			if u.DownloadTotal > 0 {
				w.dlLabel.SetText(fmt.Sprintf("%.0f%%  %.1f / %.1f MB",
					u.DownloadPct,
					float64(u.DownloadBytes)/1048576,
					float64(u.DownloadTotal)/1048576))
			} else {
				w.dlLabel.SetText(fmt.Sprintf("%.0f%%", u.DownloadPct))
			}
		})
	}

	if u.ModelReady {
		_ = w.state.ModelReady.Set(true)
		fyne.Do(func() {
			w.dlBtn.Hide()
			w.dlBar.SetValue(1)
			w.dlLabel.SetText("ready")
		})
	} else if !u.Downloading && u.State == gottApp.StateModelMissing {
		_ = w.state.ModelReady.Set(false)
		fyne.Do(func() {
			w.dlBtn.Show()
			w.dlBar.Hide()
			w.dlLabel.SetText("")
		})
	}

	switch u.State {
	case gottApp.StateRecording:
		fyne.Do(func() { w.recBtn.SetText("■ STOP") })
		_ = w.state.IsRecording.Set(true)
	case gottApp.StateIdle, gottApp.StateListening, gottApp.StateError, gottApp.StateModelMissing, gottApp.StateTranscribing:
		fyne.Do(func() { w.recBtn.SetText("● RECORD") })
		_ = w.state.IsRecording.Set(false)
	}
	if u.Hotkey != "" {
		_ = w.state.Hotkey.Set(u.Hotkey)
		fyne.Do(func() { w.hotkeyEntry.SetText(u.Hotkey) })
	}
}

func (w *Window) setStatus(s string) { w.statusLabel.SetText(s) }

func (w *Window) pollUpdates() {
	levelCh := w.a.LevelChannel()
	for {
		select {
		case <-w.pollerStop:
			return
		case lvl := <-levelCh:
			fyne.Do(func() { w.levelBar.SetValue(float64(lvl)) })
		}
	}
}

func (w *Window) toggleRecord() {
	rec, _ := w.state.IsRecording.Get()
	if rec {
		if w.cb != nil && w.cb.OnStop != nil {
			w.cb.OnStop()
		}
		return
	}
	sel, _ := w.state.SelectedMic.Get()
	if w.cb != nil && w.cb.OnStart != nil {
		w.cb.OnStart(sel)
	}
}

func (w *Window) populateMics(mics []audio.MicInfo) {
	names := make([]string, 0, len(mics))
	for _, m := range mics {
		names = append(names, m.Name)
	}
	fyne.Do(func() {
		w.micSelect.Options = names
		if len(names) > 0 {
			w.micSelect.SetSelected(names[0])
		}
	})
}

func (w *Window) Hide() { w.w.Hide() }

func (w *Window) Quit() { w.fyneApp.Quit() }

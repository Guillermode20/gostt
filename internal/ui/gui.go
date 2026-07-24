// Package ui wraps the Fyne v2 GUI around the orchestrator's Update
// channel. Every UI binding is read from the orchestrator on the Fyne
// goroutine via fyne.Do; writes back from widgets (e.g., hotkey changes)
// go through the orchestrator's WorkerMsg channel.
package ui

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/data/binding"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/layout"
	"fyne.io/fyne/v2/widget"

	"github.com/gott/gott/internal/app"
	"github.com/gott/gott/internal/audio"
	"github.com/gott/gott/internal/clipboard"
	"github.com/gott/gott/internal/settings"
)

// Callbacks lets the GUI send messages back to the orchestrator's worker.
type Callbacks struct {
	OnStart       func(deviceName string)
	OnStop        func()
	OnDownload    func()
	OnCopy        func()
	OnClear       func()
	OnSaveSettings func(s settings.Settings)
	OnUnselectMic func()
	OnQuit        func()
}

// State holds every reactive binding the GUI reads from. The orchestrator
// updates these via Fyne's data binding API (which marshals to the main
// goroutine).
type State struct {
	Status        binding.String
	InputLevel    binding.Float
	ModelReady    binding.Bool
	Hotkey        binding.String
	Threads       binding.Int
	AutoType      binding.Bool
	Transcription binding.String
	MicOptions    binding.StringList
	SelectedMic   binding.String
	ModelID       binding.String
	IsRecording   binding.Bool
}

// NewState returns a State with sensible defaults so the GUI can be shown
// before the orchestrator has fully come up.
func NewState() *State {
	list := binding.NewStringList()
	return &State{
		Status:        binding.NewString(),
		InputLevel:    binding.NewFloat(),
		ModelReady:    binding.NewBool(),
		Hotkey:        binding.NewString(),
		Threads:       binding.NewInt(),
		AutoType:      binding.NewBool(),
		Transcription: binding.NewString(),
		MicOptions:    list,
		SelectedMic:   binding.NewString(),
		ModelID:       binding.NewString(),
		IsRecording:   binding.NewBool(),
	}
}

// Window ties together the Fyne widgets.
type Window struct {
	a *app.App
	fyneApp fyne.App
	w fyne.Window

	state *State
	cb    *Callbacks

	recBtn        *widget.Button
	dlBtn         *widget.Button
	copyBtn       *widget.Button
	clearBtn      *widget.Button
	micSelect     *widget.Select
	levelBar      *widget.ProgressBar
	transcript    *widget.Entry
	statusLabel   *widget.Label
	hotkeyEntry   *widget.Entry
	threadsEntry  *widget.Entry
	autoTypeCheck *widget.Check
	helpLabel     *widget.Label
	settingsBox   *fyne.Container

	// hide-on-close pattern keeps the process alive when the user Xes out.
	wantsQuit bool

	pollerStop chan struct{}
}

// RunWindow blocks until the user quits via tray/menu/file picker. It
// spawns a single update-polling goroutine and tears everything down on
// exit.
func RunWindow(a *app.App, state *State, cb *Callbacks, title string) error {
	w := &Window{
		a:         a,
		fyneApp:   app.New(),
		state:     state,
		cb:        cb,
		pollerStop: make(chan struct{}),
	}
	// Fyne defaults to its built-in dark theme; no SetTheme call needed.
	w.w = w.fyneApp.NewWindow(title)
	w.w.Resize(fyne.NewSize(820, 640))
	w.w.SetCloseIntercept(func() { w.w.Hide() })

	w.buildUI()
	w.attachBindings()

	// Seed initial values from current settings.
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

func (w *Window) buildUI() {
	// Status bar
	w.statusLabel = widget.NewLabel("starting…")
	w.statusLabel.Wrapping = fyne.TextWrapWord
	w.helpLabel = widget.NewLabel("Hold Ctrl+Space to record, or click ● Record.")
	w.helpLabel.Wrapping = fyne.TextWrapWord

	statusBar := container.NewVBox(w.statusLabel, w.helpLabel)

	// Microphone row
	w.micSelect = widget.NewSelect([]string{}, func(name string) {
		w.cb.OnClear() // clear previous transcript on device change
		w.cb.OnSaveSettings
		// The actual recording start is triggered by the record button.
	})
	w.micSelect.PlaceHolder = "(select a microphone)"
	refreshBtn := widget.NewButton("↻", func() {
		// Re-enumerate by asking orchestrator to re-scan.
		w.cb.OnClear()
		w.cb.OnUnselectMic()
	})
	micRow := container.NewBorder(nil, nil, widget.NewLabel("Mic:"), refreshBtn, w.micSelect)

	// Level bar + record button
	w.levelBar = widget.NewProgressBar()
	w.levelBar.Min = 0
	w.levelBar.Max = 1

	w.recBtn = widget.NewButton("● Record", func() {
		w.toggleRecord()
	})
	w.recBtn.Importance = widget.HighImportance

	w.dlBtn = widget.NewButton("Download Model", func() {
		w.cb.OnDownload()
	})
	w.dlBtn.Disable()

	w.copyBtn = widget.NewButton("Copy", func() {
		s, _ := w.state.Transcription.Get()
		if s == "" {
			return
		}
		_ = clipboard.Write(s)
		fyne.Do(func() {
			w.setStatus("copied to clipboard")
		})
	})
	w.clearBtn = widget.NewButton("Clear", func() {
		w.cb.OnClear()
	})

	w.transcript = widget.NewMultiLineEntry()
	w.transcript.SetPlaceHolder("Transcription will appear here…")
	w.transcript.Disable()

	w.copyClearRow := container.NewHBox(w.copyBtn, w.clearBtn, layout.NewSpacer())

	topHalf := container.NewVBox(
		statusBar,
		micRow,
		container.NewBorder(nil, nil, nil, w.dlBtn, container.NewBorder(nil, nil, widget.NewLabel("Level"), w.recBtn, w.levelBar)),
		w.copyClearRow,
		w.transcript,
	)

	// Settings panel
	w.buildSettingsPanel()

	// Final layout
	content := container.NewBorder(nil, w.settingsBox, nil, nil, topHalf)
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
			return fmt.Errorf("threads must be 1..8")
		}
		return nil
	}
	w.threadsEntry.OnChanged = func(s string) {
		var n int
		_, _ = fmt.Sscanf(s, "%d", &n)
		_ = w.state.Threads.Set(n)
	}

	w.autoTypeCheck = widget.NewCheck("Auto-type into active window", func(on bool) {
		_ = w.state.AutoType.Set(on)
	})

	settingsForm := container.NewVBox(
		widget.NewLabel("Hotkey (e.g. ctrl+space, alt+space, f9):"),
		w.hotkeyEntry,
		widget.NewLabel("Threads (1..8):"),
		w.threadsEntry,
		w.autoTypeCheck,
		widget.NewButton("Save Settings", w.saveSettings),
	)
	settingsItem := widget.NewAccordionItem("Settings (click to expand)", settingsForm)
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
		dialog.ShowError(err, w.w)
		return
	}
	if w.cb.OnSaveSettings != nil {
		w.cb.OnSaveSettings(cfg)
	}
	w.setStatus("settings saved")
}

// attachBindings reads from the orchestrator's update channel and applies
// those to widgets on the Fyne goroutine via fyne.Do.
func (w *Window) attachBindings() {
	ch := w.a.UpdateChannel()

	go func() {
		for u := range ch {
			fyne.Do(func() {
				w.apply(u)
			})
		}
	}()

	// Initial mic population.
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

func (w *Window) apply(u app.Update) {
	if u.Status != "" {
		w.setStatus(u.Status)
	}
	if u.InputLevel >= 0 {
		_ = w.state.InputLevel.Set(float64(u.InputLevel))
		w.levelBar.SetValue(float64(u.InputLevel))
	}
	if u.Transcription != "" {
		_ = w.state.Transcription.Set(u.Transcription)
		w.transcript.SetText(u.Transcription)
	}
	if !u.ModelReady {
		_ = w.state.ModelReady.Set(false)
		w.dlBtn.Enable()
	} else {
		_ = w.state.ModelReady.Set(true)
		w.dlBtn.Disable()
	}
	switch u.State {
	case app.StateRecording:
		w.recBtn.SetText("■ Stop")
		_ = w.state.IsRecording.Set(true)
	case app.StateIdle, app.StateListening, app.StateError, app.StateModelMissing, app.StateTranscribing:
		w.recBtn.SetText("● Record")
		_ = w.state.IsRecording.Set(false)
	}
	if u.Hotkey != "" {
		_ = w.state.Hotkey.Set(u.Hotkey)
	}
}

// setStatus updates the status banner.
func (w *Window) setStatus(s string) { w.statusLabel.SetText(s) }

// pollUpdates periodically reconciles widgets from the binding API in case
// a state missed the wakeup tick.
func (w *Window) pollUpdates() {
	t := time.NewTicker(120 * time.Millisecond)
	defer t.Stop()
	for {
		select {
		case <-w.pollerStop:
			return
		case <-t.C:
			if lvl, err := w.state.InputLevel.Get(); err == nil {
				fyne.Do(func() { w.levelBar.SetValue(lvl) })
			}
		}
	}
}

// toggleRecord is the button click handler. It reflects the IsRecording
// binding so double-clicks don't cause double-starts.
func (w *Window) toggleRecord() {
	rec, _ := w.state.IsRecording.Get()
	if rec {
		w.cb.OnStop()
		return
	}
	sel, _ := w.state.SelectedMic.Get()
	if w.cb.OnStart != nil {
		w.cb.OnStart(sel)
	}
}

func (w *Window) populateMics(mics []audio.MicInfo) {
	fyne.Do(func() {
		names := make([]string, 0, len(mics))
		for _, m := range mics {
			names = append(names, m.Name)
		}
		w.micSelect.Options = names
		if len(names) > 0 {
			w.micSelect.SetSelected(names[0])
		}
	})
}

// Hide hides the window; the orchestrator keeps running so user can still
// use the global hotkey.
func (w *Window) Hide() { w.w.Hide() }

// Quit cleanly terminates the Fyne loop.
func (w *Window) Quit() { w.fyneApp.Quit() }

// (no stray locks; sync package is no longer needed here)

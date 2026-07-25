//go:build linux || darwin || freebsd

package tray

import (
	"fmt"

	"github.com/getlantern/systray"
)

// Run blocks until Quit is called. It must be called on the goroutine
// that owns the OS message loop for your process — for our purposes this
// means "the main goroutine BEFORE any other long-lived goroutines exit".
//
// A second goroutine should call Quit() when the orchestrator is shutting
// down.
func (t *Tray) Run() error {
	s := &runner{t: t}
	systray.Run(func() { s.onReady() }, func() { s.onExit() })
	return nil
}

type runner struct {
	t *Tray

	mStatus *systray.MenuItem
	mHotkey *systray.MenuItem
	mToggle *systray.MenuItem
	mQuit   *systray.MenuItem
}

func (r *runner) onReady() {
	systray.SetTitle("gostt")
	systray.SetTooltip("gostt — offline speech-to-text")
	r.mStatus = systray.AddMenuItem("Status: starting", "Application status")
	r.mStatus.Disable()
	systray.AddSeparator()
	r.mHotkey = systray.AddMenuItem("Hotkey: -", "Hold-to-talk hotkey")
	r.mHotkey.Disable()
	systray.AddSeparator()
	r.mToggle = systray.AddMenuItem("Show / Hide Window", "Toggle the gostt window")
	systray.AddSeparator()
	r.mQuit = systray.AddMenuItem("Quit gostt", "Exit the application")
	go r.pump()
}

func (r *runner) onExit() {
	// systray owns the loop and quits when r.mQuit is clicked.
}

func (r *runner) pump() {
	for {
		select {
		case st := <-r.t.updateCh:
			r.apply(st)
		case <-r.t.qCh:
			systray.Quit()
			return
		case <-r.mToggle.ClickedCh:
			if r.t.OnToggle != nil {
				r.t.OnToggle()
			}
		case <-r.mQuit.ClickedCh:
			if r.t.OnQuit != nil {
				r.t.OnQuit()
			}
			systray.Quit()
			return
		}
	}
}

func (r *runner) apply(st Status) {
	r.t.title.Store(st.Title)

	// Compose the visible title with a unicode suffix per state.
	suffix := ""
	switch st.Kind {
	case IconRecording:
		suffix = " ●"
	case IconWorking:
		suffix = " …"
	case IconError:
		suffix = " !"
	case IconModelMissing:
		suffix = " ⤓"
	}
	systray.SetTitle(st.Title + suffix)
	if st.Tooltip != "" {
		systray.SetTooltip(st.Tooltip)
	}
	if r.mStatus != nil {
		r.mStatus.SetTitle(fmt.Sprintf("Status: %s", st.Title))
	}
	if r.mHotkey != nil {
		r.mHotkey.SetTitle(fmt.Sprintf("Hotkey: %s", st.Hotkey))
	}
}

//go:build linux || darwin || freebsd

package tray

import (
	"fmt"

	"github.com/getlantern/systray"
)

// runSystrayOnce is the real systray implementation. Kept in a separate
// file with a build tag so non-Unix CI doesn't fail.
func runSystrayOnce(t *Tray) error {
	s := &trayState{t: t}
	systray.Run(func() { onReady(s) }, func() { onExit(s) })
	return nil
}

func onReady(s *trayState) {
	systray.SetTitle("gott")
	systray.SetTooltip("gott — offline speech-to-text")

	s.mStatus = wrapMenuItem(systray.AddMenuItem("Status: starting", "Application status"))
	s.mStatus.Disable()
	systray.AddSeparator()

	s.mHotkey = wrapMenuItem(systray.AddMenuItem("Hotkey: -", "Hold-to-talk hotkey"))
	s.mHotkey.Disable()
	systray.AddSeparator()

	s.mToggle = wrapMenuItem(systray.AddMenuItem("Show / Hide Window", "Toggle the gott window"))
	systray.AddSeparator()

	s.mQuit = wrapMenuItem(systray.AddMenuItem("Quit gott", "Exit the application"))

	go s.pump()
}

func onExit(s *trayState) {
	// Nothing to clean up — systray owns the loop.
}

func wrapMenuItem(m *systray.MenuItem) *menuItem {
	return &menuItem{fn: func() {
		m.ClickedCh <- struct{}{}
	}}
}

// pump forwards status updates and clicks between the systray UI thread
// and the orchestrator's Go channels.
func (s *trayState) pump() {
	apply := func(st Status) {
		s.t.title.Store(st.Title)
		// systray allows ASCII/isLetter characters only in the title.
		systray.SetTitle(st.Title)
		systray.SetTooltip(st.Tooltip)
		if s.mStatus != nil {
			label := fmt.Sprintf("Status: %s", st.Title)
			s.mStatus.fnSetTitle(label)
		}
		if s.mHotkey != nil {
			label := fmt.Sprintf("Hotkey: %s", st.Hotkey)
			s.mHotkey.fnSetTitle(label)
		}
		s.currentIcon = st.Kind
		// Icon switching is intentionally title-only so we don't have to
		// embed binary blobs:
		//   idle        => "gott"
		//   recording   => "gott ●"
		//   working     => "gott …"
		//   err         => "gott !"
		//   model-miss  => "gott ⤓"
		switch st.Kind {
		case IconRecording:
			systray.SetTitle(st.Title + " ●")
		case IconWorking:
			systray.SetTitle(st.Title + " …")
		case IconError:
			systray.SetTitle(st.Title + " !")
		case IconModelMissing:
			systray.SetTitle(st.Title + " ⤓")
		}
	}

	for {
		select {
		case st := <-s.t.updateCh:
			apply(st)
		case <-s.t.qCh:
			systray.Quit()
			return
		case <-s.mToggle.clicked():
			if s.mToggle.onClick != nil {
				s.mToggle.onClick()
			}
		case <-s.mQuit.clicked():
			if s.mQuit.onClick != nil {
				s.mQuit.onClick()
			}
			systray.Quit()
			return
		}
	}
}

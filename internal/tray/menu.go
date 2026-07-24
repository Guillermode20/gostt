package tray

// This companion file defines the menuItem facade used elsewhere in this
// package and bridges the *systray.MenuItem ClickedCh into a typed channel
// that the orchestrator can select on.

import "github.com/getlantern/systray"

type menuItem struct {
	mi        *systray.MenuItem
	clickCh   chan struct{}
	onClick   func()
	fnSetTitle func(string)
}

func (m *menuItem) clicked() <-chan struct{} {
	if m.clickCh == nil {
		m.clickCh = make(chan struct{}, 4)
		go func() {
			for range m.mi.ClickedCh {
				m.clickCh <- struct{}{}
			}
		}()
	}
	return m.clickCh
}

func (m *menuItem) SetTitle(s string) {
	if m.mi != nil {
		m.mi.SetTitle(s)
	}
}

// makeMenuItem bundles a *systray.MenuItem into our typed façade.
func makeMenuItem(mi *systray.MenuItem) *menuItem {
	return &menuItem{
		mi:          mi,
		onClick:     func() {},
		fnSetTitle:  mi.SetTitle,
		clickCh:     make(chan struct{}, 4),
	}
}

// Package tray implements the system tray icon and context menu.
//
// The tray is a long-running object that:
//
//   - displays the application title (mutated to reflect state: "gostt",
//     "gostt ● Rec", "gostt … Transcribing", etc.)
//   - exposes a context menu with: status, hotkey, toggle, quit
//   - forwards Status updates from the orchestrator to the underlying
//     notification item (only ASCII-friendly runes are used so we don't
//     have to embed binary PNG blobs).
package tray

import (
	"sync/atomic"
)

// IconKind enumerates the dynamic tray icon states. The current
// implementation maps each kind to a unicode suffix on the title so the
// tray is binary-free; if you want true icons later, add a SetIconBytes
// here and feed PNG data from assets/.
type IconKind int

const (
	IconIdle IconKind = iota
	IconRecording
	IconWorking
	IconError
	IconModelMissing
)

// Status is the high-level state broadcast by the orchestrator.
type Status struct {
	Kind      IconKind
	Title     string
	Tooltip   string
	Hotkey    string
	Recording bool
	Loading   bool
}

// Tray is the live tray instance. Start it with Run(); close it with Quit().
//
// Run blocks; spawn it on its own goroutine unless you have nothing else
// to do.
type Tray struct {
	updateCh chan Status
	qCh      chan struct{}
	title    atomic.Value // string — last known rendered title (for tests)

	// OnToggle is invoked when the user clicks "Show / Hide Window".
	OnToggle func()
	// OnQuit is invoked when the user clicks "Quit" in the tray menu BEFORE
	// systray itself tears down, so the orchestrator can do a graceful
	// shutdown.
	OnQuit func()
}

// New returns a Tray ready to be started.
func New() *Tray {
	return &Tray{
		updateCh: make(chan Status, 8),
		qCh:      make(chan struct{}),
	}
}

// Update posts a new state. Safe to call from any goroutine.
func (t *Tray) Update(s Status) {
	select {
	case t.updateCh <- s:
	default:
		// drop oldest then enqueue
		select {
		case <-t.updateCh:
		default:
		}
		select {
		case t.updateCh <- s:
		default:
		}
	}
}

// Quit terminates Run(). Safe to call from any goroutine.
func (t *Tray) Quit() {
	select {
	case <-t.qCh:
	default:
		close(t.qCh)
	}
}

// LastTitle returns the most recent title the tray displayed. Useful for
// tests that want to verify state changes were applied.
func (t *Tray) LastTitle() string {
	v, _ := t.title.Load().(string)
	return v
}

// OnToggleHandler is invoked by the systray pump when the user clicks
// the "Show / Hide Window" entry. Returns immediately.
func (t *Tray) OnToggleHandler() func() { return t.OnToggle }

// OnQuitHandler is invoked by the systray pump before the loop terminates.
func (t *Tray) OnQuitHandler() func() { return t.OnQuit }

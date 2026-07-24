// Package hotkey implements the global hold-to-talk hotkey listener on top
// of the Freedesktop XDG GlobalShortcuts Portal. Because the portal runs
// out-of-process and *every* method call returns asynchronously via a
// Response signal, this file is mostly careful D-Bus plumbing.
package hotkey

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/godbus/dbus/v5"
)

// Event is a single hotkey state transition.
type Event int

const (
	Unknown Event = iota
	Pressed
	Released
)

func (e Event) String() string {
	switch e {
	case Pressed:
		return "Pressed"
	case Released:
		return "Released"
	default:
		return "Unknown"
	}
}

// Listener registers a single shortcut with the XDG GlobalShortcuts Portal
// and forwards Activated/Deactivated signals to a Go channel.
type Listener struct {
	conn        *dbus.Conn
	sessionPath dbus.ObjectPath

	trigger string

	events chan Event
	status chan string
	closed chan struct{}
	once   sync.Once
}

// New creates and binds a listener. The trigger is a chord string like
// "ctrl+space", "alt+space", "super+space", "f9".
func New(trigger string) (*Listener, error) {
	if trigger == "" {
		trigger = "ctrl+space"
	}
	l := &Listener{
		trigger: trigger,
		events:  make(chan Event, 8),
		status:  make(chan string, 4),
		closed:  make(chan struct{}),
	}
	if err := l.bind(); err != nil {
		return nil, err
	}
	return l, nil
}

// Events is the read end of the activation/deactivation stream.
func (l *Listener) Events() <-chan Event { return l.events }

// Status receives human-readable status messages suitable for the UI.
func (l *Listener) Status() <-chan string { return l.status }

// Close unbinds the session. Idempotent.
func (l *Listener) Close() error {
	var err error
	l.once.Do(func() {
		close(l.closed)
		if l.conn != nil && l.sessionPath != "" {
			// Best-effort: ask the portal to release the session.
			obj := l.conn.Object("org.freedesktop.portal.Desktop", dbus.ObjectPath("/org/freedesktop/portal/desktop"))
			call := obj.Call("org.freedesktop.portal.GlobalShortcuts.Close", 0, l.sessionPath)
			if call.Err != nil {
				err = fmt.Errorf("close session: %w", call.Err)
			}
		}
		if l.conn != nil {
			_ = l.conn.Close()
		}
	})
	return err
}

// bind performs CreateSession -> wait for Response -> BindShortcuts ->
// wait for Response -> subscribe to Activated/Deactivated signals.
func (l *Listener) bind() error {
	conn, err := dbus.ConnectSessionBus()
	if err != nil {
		return fmt.Errorf("session bus: %w", err)
	}
	l.conn = conn

	root := conn.Object("org.freedesktop.portal.Desktop", dbus.ObjectPath("/org/freedesktop/portal/desktop"))

	// 1. CreateSession ------------------------------------------------------------
	sessionOptions := map[string]dbus.Variant{
		"handle_token": dbus.MakeVariant(randomToken()),
	}
	l.pushStatus("creating portal session")
	createCall := root.Call(
		"org.freedesktop.portal.GlobalShortcuts.CreateSession",
		0,
		sessionOptions,
	)
	if createCall.Err != nil {
		_ = conn.Close()
		return fmt.Errorf("CreateSession: %w", createCall.Err)
	}
	requestPath := createCall.Body[0].(dbus.ObjectPath)

	sessionHandle, err := l.awaitSessionResponse(requestPath)
	if err != nil {
		_ = conn.Close()
		return fmt.Errorf("await CreateSession response: %w", err)
	}
	l.sessionPath = sessionHandle
	l.pushStatus("portal session ready")

	// 2. BindShortcuts ------------------------------------------------------------
	portalTrigger := linuxizeChord(l.trigger)
	shortcut := map[string]dbus.Variant{
		"description":         dbus.MakeVariant("Hold to talk"),
		"preferred_trigger":   dbus.MakeVariant(portalTrigger),
	}
	bindOptions := map[string]dbus.Variant{
		"handle_token": dbus.MakeVariant(randomToken()),
	}
	bindCall := root.Call(
		"org.freedesktop.portal.GlobalShortcuts.BindShortcuts",
		0,
		sessionHandle,
		[]map[string]dbus.Variant{shortcut},
		"" /* parent_window */,
		bindOptions,
	)
	if bindCall.Err != nil {
		_ = l.Close()
		return fmt.Errorf("BindShortcuts: %w", bindCall.Err)
	}
	bindRequest := bindCall.Body[0].(dbus.ObjectPath)
	if err := l.awaitBindResponse(bindRequest); err != nil {
		_ = l.Close()
		return fmt.Errorf("await BindShortcuts response: %w", err)
	}
	l.pushStatus(fmt.Sprintf("hotkey bound: %s", portalTrigger))

	// 3. Subscribe to Activated / Deactivated ------------------------------------
	if err := conn.AddMatchSignal(
		dbus.WithMatchObjectPath(sessionHandle),
		dbus.WithMatchInterface("org.freedesktop.portal.GlobalShortcuts"),
		dbus.WithMatchMember("Activated"),
	); err != nil {
		_ = l.Close()
		return fmt.Errorf("subscribe Activated: %w", err)
	}
	if err := conn.AddMatchSignal(
		dbus.WithMatchObjectPath(sessionHandle),
		dbus.WithMatchInterface("org.freedesktop.portal.GlobalShortcuts"),
		dbus.WithMatchMember("Deactivated"),
	); err != nil {
		_ = l.Close()
		return fmt.Errorf("subscribe Deactivated: %w", err)
	}

	signals := make(chan *dbus.Signal, 8)
	conn.Signal(signals)

	go l.pump(signals)
	return nil
}

// pump forwards D-Bus signals to the Go event channel until Close().
func (l *Listener) pump(signals <-chan *dbus.Signal) {
	for {
		select {
		case <-l.closed:
			return
		case s, ok := <-signals:
			if !ok {
				return
			}
			if s == nil {
				continue
			}
			switch s.Name {
			case "org.freedesktop.portal.GlobalShortcuts.Activated":
				select {
				case l.events <- Pressed:
				default:
				}
				l.pushStatus("hotkey pressed")
			case "org.freedesktop.portal.GlobalShortcuts.Deactivated":
				select {
				case l.events <- Released:
				default:
				}
				l.pushStatus("hotkey released")
			default:
				// ignore other signals
			}
		}
	}
}

// awaitSessionResponse subscribes to the Response signal on the given
// request path and waits for a session_handle result. Returns the
// session_handle object path.
func (l *Listener) awaitSessionResponse(req dbus.ObjectPath) (dbus.ObjectPath, error) {
	rule := "type='signal',interface='org.freedesktop.portal.Request',member='Response',path='" + req + "'"
	if err := l.conn.BusObject().Call("org.freedesktop.DBus.AddMatch", 0, rule).Err; err != nil {
		return "", fmt.Errorf("AddMatch: %w", err)
	}
	defer func() {
		_ = l.conn.BusObject().Call("org.freedesktop.DBus.RemoveMatch", 0, rule).Err
	}()

	ch := make(chan *dbus.Signal, 1)
	l.conn.Signal(ch)
	defer l.conn.RemoveSignal(ch)

	timeout := time.NewTimer(8 * time.Second)
	defer timeout.Stop()

	for {
		select {
		case <-l.closed:
			return "", errors.New("listener closed")
		case <-timeout.C:
			return "", errors.New("timed out waiting for CreateSession response")
		case s := <-ch:
			if s == nil {
				continue
			}
			if len(s.Body) < 2 {
				continue
			}
			results, ok := s.Body[1].(map[string]dbus.Variant)
			if !ok {
				continue
			}
			v, present := results["session_handle"]
			if !present {
				return "", errors.New("session_handle missing in CreateSession response")
			}
			return v.Value().(dbus.ObjectPath), nil
		}
	}
}

// awaitBindResponse is the BindShortcuts counterpart. We only need to
// confirm success; we don't read individual shortcut state here.
func (l *Listener) awaitBindResponse(req dbus.ObjectPath) error {
	rule := "type='signal',interface='org.freedesktop.portal.Request',member='Response',path='" + req + "'"
	if err := l.conn.BusObject().Call("org.freedesktop.DBus.AddMatch", 0, rule).Err; err != nil {
		return fmt.Errorf("AddMatch: %w", err)
	}
	defer func() {
		_ = l.conn.BusObject().Call("org.freedesktop.DBus.RemoveMatch", 0, rule).Err
	}()
	ch := make(chan *dbus.Signal, 1)
	l.conn.Signal(ch)
	defer l.conn.RemoveSignal(ch)
	timeout := time.NewTimer(8 * time.Second)
	defer timeout.Stop()
	for {
		select {
		case <-l.closed:
			return errors.New("listener closed")
		case <-timeout.C:
			return errors.New("timed out waiting for BindShortcuts response")
		case s := <-ch:
			if s == nil {
				continue
			}
			if len(s.Body) < 2 {
				continue
			}
			results, ok := s.Body[1].(map[string]dbus.Variant)
			if !ok {
				continue
			}
			if v, present := results["shortcuts"]; present && v.Signature().String() == "a(sssusasx)" {
				// success — shortcut details were echoed back
				return nil
			}
			// Empty body with no error means success too.
			return nil
		}
	}
}

func (l *Listener) pushStatus(s string) {
	select {
	case l.status <- s:
	default:
	}
}

// linuxizeChord converts "ctrl+space" -> "Control+space", "super+space" ->
// "Super+space" etc. The XDG portal expects "<Mod>+<Key>" with
// Mod in the form Control/Shift/Alt/Super (or Meta) and keys in lowercase
// except for special keys (Space, Tab, F1, Return …).
func linuxizeChord(s string) string {
	parts := strings.Split(strings.ToLower(s), "+")
	if len(parts) == 0 {
		return s
	}
	for i, p := range parts {
		if i == len(parts)-1 {
			parts[i] = prettyKey(p)
			continue
		}
		switch p {
		case "ctrl", "control":
			parts[i] = "Control"
		case "shift":
			parts[i] = "Shift"
		case "alt":
			parts[i] = "Alt"
		case "super", "cmd", "win":
			parts[i] = "Super"
		case "meta":
			parts[i] = "Meta"
		default:
			parts[i] = p
		}
	}
	return strings.Join(parts, "+")
}

func prettyKey(k string) string {
	switch k {
	case "space", "spacebar":
		return "Space"
	case "tab":
		return "Tab"
	case "enter", "return":
		return "Return"
	case "esc", "escape":
		return "Escape"
	case "backspace":
		return "BackSpace"
	}
	if strings.HasPrefix(k, "f") && len(k) > 1 {
		// F1..F35 — pretty form is "F1".
		return strings.ToUpper(k[:1]) + k[1:]
	}
	return strings.Title(k)
}

// randomToken returns a unique-per-process handle token. The portal uses
// this only to correlate logs; any unique-ish low-collision string works.
func randomToken() string {
	const charset = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, 16)
	for i := range b {
		b[i] = charset[time.Now().UnixNano()%int64(len(charset))]
		time.Sleep(time.Nanosecond) // jitter
	}
	return "gott_" + string(b)
}

// env is used only so that debug codepaths can branch on $GOTT_DEBUG.
var _ = func() bool { return os.Getenv("GOTT_DEBUG") != "" }

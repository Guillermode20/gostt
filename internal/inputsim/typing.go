// Package inputsim types text into the currently-focused window.
//
// Strategy: prefer the per-session native tool:
//   - X11        -> xdotool type --clearmodifiers
//   - Wayland    -> ydotool type (preferred) or wtype
//   - Fallback   -> talk to a virtual uinput keyboard (advanced users only)
//
// All implementations write "chunks" so that very long dictation outputs
// don't overrun command-line argument limits.
package inputsim

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"sync"
	"time"

	clipboardx "github.com/gott/gott/internal/clipboard"

	"github.com/bendahl/uinput"
)

// TypeOptions controls timing and chunking.
type TypeOptions struct {
	// ChunkSize is the largest -argument (or chunked payload) we'll pass
	// per invocation. Command-line length limits vary but 200 chars is a
	// universally safe value on Linux.
	ChunkSize int
	// Delay is inter-chunk delay for tooling that needs it.
	Delay time.Duration
}

func defaultOptions() TypeOptions {
	return TypeOptions{
		ChunkSize: 200,
		Delay:     15 * time.Millisecond,
	}
}

// Backend identifies which linux typing implementation was selected.
type Backend string

const (
	BackendXdotool   Backend = "xdotool"
	BackendYdotool   Backend = "ydotool"
	BackendWtype     Backend = "wtype"
	BackendUinput    Backend = "uinput"
	BackendClipboard Backend = "clipboard-only"
)

// TypeResult is returned to the caller so it can show "typed via X" in the
// status line.
type TypeResult struct {
	Backend Backend
	Chunks  int
}

// TypeText types s into the active window using the best available
// strategy for the current session.
func TypeText(s string) (TypeResult, error) {
	return TypeTextWith(s, defaultOptions())
}

// TypeTextWith is TypeText with caller-supplied options.
func TypeTextWith(s string, opts TypeOptions) (TypeResult, error) {
	if s == "" {
		return TypeResult{}, errors.New("empty input")
	}
	if opts.ChunkSize <= 0 {
		opts.ChunkSize = 200
	}

	session := detectSession()
	switch session {
	case "x11":
		if _, err := exec.LookPath("xdotool"); err == nil {
			if err := runXdotool(s, opts); err == nil {
				return TypeResult{Backend: BackendXdotool}, nil
			} else {
				return TypeResult{}, err
			}
		}
	case "wayland":
		if _, err := exec.LookPath("ydotool"); err == nil {
			if err := runYdotool(s, opts); err == nil {
				return TypeResult{Backend: BackendYdotool}, nil
			} else {
				return TypeResult{}, err
			}
		}
		if _, err := exec.LookPath("wtype"); err == nil {
			if err := runWtype(s, opts); err == nil {
				return TypeResult{Backend: BackendWtype}, nil
			} else {
				return TypeResult{}, err
			}
		}
	}
	// No backend available? Copy to clipboard so the user can paste manually.
	if err := writeClipboard(s); err != nil {
		return TypeResult{}, fmt.Errorf("no typing backend available and clipboard write failed: %w", err)
	}
	return TypeResult{Backend: BackendClipboard}, nil
}

func detectSession() string {
	if t := os.Getenv("XDG_SESSION_TYPE"); t != "" {
		return t
	}
	if os.Getenv("WAYLAND_DISPLAY") != "" {
		return "wayland"
	}
	if os.Getenv("DISPLAY") != "" {
		return "x11"
	}
	return ""
}

func chunk(s string, n int) []string {
	if len(s) <= n {
		return []string{s}
	}
	var out []string
	for i := 0; i < len(s); i += n {
		end := i + n
		if end > len(s) {
			end = len(s)
		}
		out = append(out, s[i:end])
	}
	return out
}

// runXdotool uses xdotool type --clearmodifiers in chunks.
func runXdotool(s string, opts TypeOptions) error {
	chunks := chunk(s, opts.ChunkSize)
	for i, c := range chunks {
		cmd := exec.Command("xdotool", "type", "--clearmodifiers", "--", c)
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("xdotool chunk %d: %w", i, err)
		}
		if opts.Delay > 0 && i < len(chunks)-1 {
			time.Sleep(opts.Delay)
		}
	}
	return nil
}

// runYdotool delegates to ydotool (it expects types via the ydotoold daemon).
func runYdotool(s string, opts TypeOptions) error {
	chunks := chunk(s, opts.ChunkSize)
	for i, c := range chunks {
		cmd := exec.Command("ydotool", "type", "--", c)
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("ydotool chunk %d: %w", i, err)
		}
		if opts.Delay > 0 && i < len(chunks)-1 {
			time.Sleep(opts.Delay)
		}
	}
	return nil
}

// runWtype handles wtype on Wayland.
func runWtype(s string, opts TypeOptions) error {
	chunks := chunk(s, opts.ChunkSize)
	for i, c := range chunks {
		cmd := exec.Command("wtype", "--", c)
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("wtype chunk %d: %w", i, err)
		}
		if opts.Delay > 0 && i < len(chunks)-1 {
			time.Sleep(opts.Delay)
		}
	}
	return nil
}

// writeClipboard uses our internal clipboard wrapper so the user can paste
// manually when no typing backend is available.
func writeClipboard(s string) error {
	return clipboardx.Write(s)
}

// ----------------------------------------------------------------------------
// Uinput virtual keyboard (power-user fallback)
// ----------------------------------------------------------------------------

// UinputTyper is a long-lived virtual keyboard device used for low-level
// keystroke injection. Most users won't need this — xdotool / wtype /
// ydotool are faster and don't require /dev/uinput write access.
//
// IMPORTANT: full Unicode text typing via uinput requires building a
// translation table; we only support ASCII printable and a small list of
// common punctuation marks here. Anything outside that set is rejected
// so we don't silently emit garbage.
type UinputTyper struct {
	once    sync.Once
	keyboard uinput.Keyboard
}

// NewUinput opens /dev/uinput and creates a virtual keyboard device.
// Caller must Close() it.
func NewUinput() (*UinputTyper, error) {
	t := &UinputTyper{}
	k, err := uinput.CreateKeyboard("/dev/uinput", []byte("gott-virtual-keyboard"))
	if err != nil {
		return nil, fmt.Errorf("create uinput keyboard (need /dev/uinput RW access): %w", err)
	}
	t.keyboard = k
	return t, nil
}

// Type sends keystrokes for s, returning an error if any rune is outside
// the supported ASCII range.
func (u *UinputTyper) Type(s string) error {
	if u.keyboard == nil {
		return errors.New("uinput keyboard not initialised")
	}
	for _, r := range s {
		if err := u.sendRune(r); err != nil {
			return err
		}
		time.Sleep(3 * time.Millisecond)
	}
	return nil
}

// Close destroys the virtual keyboard device.
func (u *UinputTyper) Close() error {
	var err error
	u.once.Do(func() {
		if u.keyboard != nil {
			err = u.keyboard.Close()
		}
	})
	return err
}

func (u *UinputTyper) sendRune(r rune) error {
	if r > 0x7e {
		return fmt.Errorf("uinput backend does not support non-ASCII rune %q", r)
	}
	switch {
	case r >= 'a' && r <= 'z':
		return u.keyboard.KeyPress(letterKeys[r-'a'])
	case r >= 'A' && r <= 'Z':
		if err := u.keyboard.KeyDown(uinput.KeyLeftshift); err != nil {
			return err
		}
		err := u.keyboard.KeyPress(letterKeys[r-'A'])
		_ = u.keyboard.KeyUp(uinput.KeyLeftshift)
		return err
	case r >= '1' && r <= '9':
		return u.keyboard.KeyPress(digitKeys[r-'1'])
	case r == '0':
		return u.keyboard.KeyPress(uinput.Key0)
	case r == ' ':
		return u.keyboard.KeyPress(uinput.KeySpace)
	case r == '\n':
		return u.keyboard.KeyPress(uinput.KeyEnter)
	case r == '\t':
		return u.keyboard.KeyPress(uinput.KeyTab)
	}
	// Punctuation
	if k, ok := punctKeys[r]; ok {
		return u.keyboard.KeyPress(k)
	}
	// Shifted punctuation
	if k, ok := shiftedPunctKeys[r]; ok {
		if err := u.keyboard.KeyDown(uinput.KeyLeftshift); err != nil {
			return err
		}
		err := u.keyboard.KeyPress(k)
		_ = u.keyboard.KeyUp(uinput.KeyLeftshift)
		return err
	}
	return fmt.Errorf("unsupported rune %q", r)
}

// Letter keys a..z (lowercase, no shift).
var letterKeys = []int{
	uinput.KeyA, uinput.KeyB, uinput.KeyC, uinput.KeyD, uinput.KeyE,
	uinput.KeyF, uinput.KeyG, uinput.KeyH, uinput.KeyI, uinput.KeyJ,
	uinput.KeyK, uinput.KeyL, uinput.KeyM, uinput.KeyN, uinput.KeyO,
	uinput.KeyP, uinput.KeyQ, uinput.KeyR, uinput.KeyS, uinput.KeyT,
	uinput.KeyU, uinput.KeyV, uinput.KeyW, uinput.KeyX, uinput.KeyY,
	uinput.KeyZ,
}

// Digit keys 1..9.
var digitKeys = []int{
	uinput.Key1, uinput.Key2, uinput.Key3, uinput.Key4, uinput.Key5,
	uinput.Key6, uinput.Key7, uinput.Key8, uinput.Key9,
}

// Punctuation keys (no shift).
var punctKeys = map[rune]int{
	'-': uinput.KeyMinus,
	'=': uinput.KeyEqual,
	'[': uinput.KeyLeftbrace,
	']': uinput.KeyRightbrace,
	';': uinput.KeySemicolon,
	'`': uinput.KeyGrave,
	'\\': uinput.KeyBackslash,
	',': uinput.KeyComma,
	'.': uinput.KeyDot,
	'/': uinput.KeySlash,
}

// Shifted punctuation. Requires shift held.
var shiftedPunctKeys = map[rune]int{
	'_': uinput.KeyMinus,
	'+': uinput.KeyEqual,
	'{': uinput.KeyLeftbrace,
	'}': uinput.KeyRightbrace,
	':': uinput.KeySemicolon,
	'~': uinput.KeyGrave,
	'|': uinput.KeyBackslash,
	'<': uinput.KeyComma,
	'>': uinput.KeyDot,
	'?': uinput.KeySlash,
	'!': uinput.Key1,
	'@': uinput.Key2,
	'#': uinput.Key3,
	'$': uinput.Key4,
	'%': uinput.Key5,
	'^': uinput.Key6,
	'&': uinput.Key7,
	'*': uinput.Key8,
	'(': uinput.Key9,
	')': uinput.Key0,
	'"': uinput.KeyApostrophe,
	'\'': uinput.KeyApostrophe,
}


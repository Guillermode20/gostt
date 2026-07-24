// Package settings loads and persists user configuration to the XDG config
// directory (~/.config/gott/config.json).
//
// The package is intentionally small and dependency-free so it can be used
// safely from any layer (CLI, GUI, worker goroutines) without taking locks.
package settings

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
)

// Default hold-to-talk shortcut. Matches the rustt default.
const DefaultHotkey = "ctrl+space"

// Default STT model key.
const DefaultModel = "parakeet-tdt-int8"

// Settings represents the persisted user configuration.
//
// Every field is safe to mutate concurrently, but Save() should typically be
// called from a single goroutine to avoid competing writes to disk.
type Settings struct {
	// Model is the active model id (see internal/transcription/model.go).
	Model string `json:"model"`

	// Language is the two-letter ISO 639-1 code (only "en" is currently wired).
	Language string `json:"language"`

	// Threads is the number of CPU threads used by the inference engine. It is
	// clamped to a maximum of 4 to mirror rustt behaviour so we don't fight
	// the user's power profile for no measurable accuracy gain.
	Threads int `json:"threads"`

	// PreferredDevice is an optional microphone name. If nil the system
	// default capture device is used.
	PreferredDevice *string `json:"preferred_device,omitempty"`

	// HoldToTalkKey is a chord string e.g. "ctrl+space", "alt+space", "f9".
	HoldToTalkKey string `json:"hold_to_talk_key"`

	// AutoType controls whether the transcribed text is auto-typed into the
	// active window in addition to being copied to the clipboard.
	AutoType bool `json:"auto_type"`

	// AutoPaste is the legacy rustt alias for AutoType. Load populates
	// AutoType from it when present and AutoType has not been explicitly set.
	AutoPaste *bool `json:"auto_paste,omitempty"`
}

// DefaultSettings returns sane defaults computed at call time. Thread count
// is clamped to 4 to mirror rustt behaviour.
func DefaultSettings() Settings {
	threads := runtime.NumCPU()
	if threads <= 0 {
		threads = 1
	}
	if threads > 4 {
		threads = 4
	}
	return Settings{
		Model:         DefaultModel,
		Language:      "en",
		Threads:       threads,
		PreferredDevice: nil,
		HoldToTalkKey: DefaultHotkey,
		AutoType:      true,
	}
}

// ConfigPath returns the XDG-resolved on-disk path of the config file. The
// directory is created if missing. Honours $XDG_CONFIG_HOME if set.
func ConfigPath() (string, error) {
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	cfgDir := filepath.Join(dir, "gott")
	if err := os.MkdirAll(cfgDir, 0o755); err != nil {
		return "", err
	}
	return filepath.Join(cfgDir, "config.json"), nil
}

// Load reads config.json, applies defaults where fields are missing, and
// honours the legacy "auto_paste" alias.
//
// If the file does not exist, DefaultSettings is returned (no error). If the
// file exists but is malformed, the underlying error is returned so the
// caller can decide whether to fall back to defaults or surface the problem
// to the user.
func Load() (Settings, error) {
	path, err := ConfigPath()
	if err != nil {
		return Settings{}, err
	}
	cfg := DefaultSettings()

	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return cfg, nil
		}
		return Settings{}, err
	}

	// Decode into a permissive map first so we can detect missing fields and
	// so the legacy "auto_paste" alias can win when AutoType is absent.
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return Settings{}, err
	}

	get := func(key string) (json.RawMessage, bool) {
		v, ok := raw[key]
		return v, ok
	}
	if v, ok := get("model"); ok {
		_ = json.Unmarshal(v, &cfg.Model)
	}
	if v, ok := get("language"); ok {
		_ = json.Unmarshal(v, &cfg.Language)
	}
	if v, ok := get("threads"); ok {
		var t int
		if err := json.Unmarshal(v, &t); err == nil && t > 0 {
			cfg.Threads = t
		}
	}
	if v, ok := get("preferred_device"); ok && len(v) > 0 && string(v) != "null" {
		var s string
		if err := json.Unmarshal(v, &s); err == nil {
			cfg.PreferredDevice = &s
		}
	}
	if v, ok := get("hold_to_talk_key"); ok {
		_ = json.Unmarshal(v, &cfg.HoldToTalkKey)
	}
	if v, ok := get("auto_type"); ok {
		_ = json.Unmarshal(v, &cfg.AutoType)
	}
	if v, ok := get("auto_paste"); ok && len(v) > 0 && string(v) != "null" {
		var b bool
		if err := json.Unmarshal(v, &b); err == nil {
			cfg.AutoPaste = &b
			if _, explicit := get("auto_type"); !explicit {
				cfg.AutoType = b
			}
		}
	}

	// Validate / clamp.
	if cfg.HoldToTalkKey == "" {
		cfg.HoldToTalkKey = DefaultHotkey
	}
	if cfg.Threads <= 0 {
		cfg.Threads = 1
	}
	if cfg.Threads > 4 {
		cfg.Threads = 4
	}
	if cfg.Model == "" {
		cfg.Model = DefaultModel
	}
	if cfg.Language == "" {
		cfg.Language = "en"
	}
	return cfg, nil
}

// Save atomically writes settings (write-temp + rename) so a process kill
// mid-write can never leave a half-written JSON file.
func Save(s Settings) error {
	path, err := ConfigPath()
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// AutoTypeEnabled reports whether the effective auto-type flag is on.
// Provided so call sites don't have to second-guess the legacy alias.
func (s Settings) AutoTypeEnabled() bool {
	return s.AutoType
}

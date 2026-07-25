package settings

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultSettings(t *testing.T) {
	s := DefaultSettings()
	if s.Model != DefaultModel {
		t.Errorf("default model = %q, want %q", s.Model, DefaultModel)
	}
	if s.HoldToTalkKey != DefaultHotkey {
		t.Errorf("default hotkey = %q, want %q", s.HoldToTalkKey, DefaultHotkey)
	}
	if s.Language != "en" {
		t.Errorf("default language = %q, want en", s.Language)
	}
	if s.Threads <= 0 || s.Threads > 4 {
		t.Errorf("default threads = %d, want 1..4", s.Threads)
	}
	if !s.AutoType {
		t.Errorf("default auto_type = false, want true")
	}
}

func TestSaveLoadRoundtrip(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", tmp)

	original := DefaultSettings()
	original.HoldToTalkKey = "alt+space"
	original.PreferredDevice = stringPtr("USB Microphone")
	original.AutoType = false

	if err := Save(original); err != nil {
		t.Fatalf("Save: %v", err)
	}

	got, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got.HoldToTalkKey != "alt+space" {
		t.Errorf("HoldToTalkKey = %q, want alt+space", got.HoldToTalkKey)
	}
	if got.PreferredDevice == nil || *got.PreferredDevice != "USB Microphone" {
		t.Errorf("PreferredDevice = %v, want USB Microphone", got.PreferredDevice)
	}
	if got.AutoType {
		t.Errorf("AutoType = true, want false")
	}
}

func TestLoadLegacyAutoPaste(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", tmp)
	cfgDir := filepath.Join(tmp, "gostt")
	if err := os.MkdirAll(cfgDir, 0o755); err != nil {
		t.Fatal(err)
	}
	legacy := map[string]any{
		"model":           "parakeet-tdt-int8",
		"language":        "en",
		"threads":         2,
		"hold_to_talk_key": "ctrl+space",
		"auto_paste":      false, // legacy field only
	}
	b, _ := json.MarshalIndent(legacy, "", "  ")
	if err := os.WriteFile(filepath.Join(cfgDir, "config.json"), b, 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got.AutoType {
		t.Errorf("AutoType=true after legacy auto_paste=false; expected false")
	}
}

func TestMissingFileReturnsDefaults(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", tmp)
	got, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got.Model != DefaultModel {
		t.Errorf("Model = %q, want default %q", got.Model, DefaultModel)
	}
}

func stringPtr(s string) *string { return &s }

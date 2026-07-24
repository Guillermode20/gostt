# gott — Offline Linux Speech-to-Text Dictation (Go)

A Go port of the `rustt` project: a privacy-focused, offline hold-to-talk STT dictation
application for Linux desktops.

## Features

- Hold-to-talk global hotkey (XDG GlobalShortcuts Portal — works on KDE, GNOME, Sway,
  Hyprland, Xfce without root).
- Offline inference via **ONNX Runtime** with the **NVIDIA Parakeet-TDT 0.6B INT8** model
  (~670 MB, 20–50 ms inference).
- Audio capture via miniaudio (PipeWire / PulseAudio / ALSA), stereo→mono downmix,
  16 kHz resampling, RMS silence trimming.
- System tray (StatusNotifierItem) with dynamic mic / recording icons.
- Direct typing into the active window (X11 via `robotgo` and `xdotool`; Wayland via
  `wtype` or virtual uinput keyboard).
- Automatic clipboard copy of transcribed text.
- Dual mode:
  - GUI: tray + Fyne window
  - CLI: `list`, `download`, `record [seconds]`

## Project Layout

```
cmd/gott/main.go             — CLI entrypoint
internal/app/app.go          — orchestrator: channels, worker loop, glue
internal/audio/capture.go    — mic enumeration + recording via malgo
internal/audio/resample.go   — mono downmix, 16 kHz resample, silence trim
internal/transcription/model.go     — model metadata, HTTP downloader
internal/transcription/engine.go    — Parakeet-TDT ONNX inference loop
internal/hotkey/portal.go    — XDG GlobalShortcuts Portal (godbus)
internal/tray/tray.go        — systray menu + dynamic icons
internal/inputsim/typing.go  — auto-typing via robotgo/xdotool/wtype/uinput
internal/settings/config.go  — XDG config (~/.config/gott/config.json)
internal/ui/gui.go           — Fyne window, controls, level bar
```

## Build & Run

The fastest reproduction path uses a [distrobox](https://distrobox.org/)
dev container — Ubuntu 24.04 with all gott's system libs, ONNX Runtime,
Go 1.23 and the project mounted in. Two manifest formats are shipped:

- `distrobox.ini` — declarative multi-target manifest; the `[target.gott-dev]`
  stanza is the developer image.
- `Dockerfile.gott-dev` — same image as a vanilla `docker build` target.

### Option A — distrobox (recommended)

```bash
# Once per host:
distrobox assemble create --file distrobox.ini

# Then enter with audio + ONNX runtime wired up:
./scripts/dev-shell.sh
```

Inside the box:

```bash
go mod tidy
go build -o gott ./cmd/gott
./gott list
./gott download          # ~670 MB on first run
./gott record 5          # 5 seconds on the system default mic
# or just `gott` for the GUI + tray
```

### Option B — vanilla Docker

```bash
docker build -f Dockerfile.gott-dev -t gott-dev .
docker run -it --rm \
    --device /dev/snd --device /dev/uinput \
    --group-add audio --group-add input \
    -v "$PWD":/go/src/github.com/gott/gott \
    -e LD_LIBRARY_PATH=/opt/onnxruntime-linux-x64-1.18.0/lib \
    gott-dev bash
```

### Option C — bare-metal Debian/Ubuntu

Install the system deps, ONNX Runtime, then `go build` as usual:

```bash
sudo apt install -y \
  libasound2-dev libgtk-3-dev libayatana-appindicator3-dev \
  libxdo-dev libxtst-dev libvulkan1 pkg-config
```

### ONNX Runtime

`onnxruntime_go` requires the matching `libonnxruntime.so` available at runtime.

```bash
# Example: download v1.18.0 CPU build
curl -L -o /tmp/onnxruntime.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
tar -xzf /tmp/onnxruntime.tgz -C /opt
sudo ln -sf /opt/onnxruntime-linux-x64-1.18.0/lib/libonnxruntime.so \
  /usr/local/lib/libonnxruntime.so
sudo ldconfig
export LD_LIBRARY_PATH=/opt/onnxruntime-linux-x64-1.18.0/lib:$LD_LIBRARY_PATH
```

### Build

```bash
go mod tidy
go build -o gott ./cmd/gott
```

### Usage

```bash
gott                       # launch GUI + tray
gott list                  # print available microphones
gott download              # download the default Parakeet-TDT INT8 model
gott record [seconds]      # record N seconds and print transcription + timings
gott --help                # usage info
```

The model is stored in `~/.local/share/gott/models/parakeet-tdt-int8/`. Config lives
in `~/.config/gott/config.json`.

## Hotkey

Default hold-to-talk hotkey: **Ctrl+Space**. Re-bindable through the in-app settings
panel or by editing `config.json`.

## Notes on ONNX inference

The Parakeet-TDT runtime is implemented around:

1. Log-mel feature extraction (80-dim, 25 ms window / 10 ms hop, Hann window).
2. Encoder pass over the full utterance → `encoder_out` `[1, T, D]`.
3. TDT joint network loop: feed `(encoder_frame[t], state)` → emit token + duration;
   advance `t` by the duration; stop when N consecutive blanks are produced (or max
   length reached).
4. Token-text mapping via `vocab.txt`.

The implementation makes assumptions about the model’s I/O names (commonly
`audio_signal`, `targets`, `target_length` on older NeMo exports, or `input_signal`
on newer ones) — see `internal/transcription/engine.go` for the exact names and how
to override them via environment variable `GOTT_TDT_INPUT_NAME` etc.

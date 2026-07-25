# gostt Makefile
#
# Quick start (host, no distrobox):
#   brew install onnxruntime    # one-time
#   make run                    # build + download model + record 5s
#
# Targets:
#   make build                  compile ./gostt
#   make download               fetch the Parakeet-TDT model (~670 MB)
#   make record [SECS=5]        record and transcribe (default 5s)
#   make run                    build + download + record
#   make clean                  remove ./gostt
#
# CGO needs libonnxruntime — the Makefile auto-discovers it via
# `brew --prefix onnxruntime` if brew is installed, otherwise falls
# back to pkg-config.

BREW      := $(shell which brew 2>/dev/null || echo /home/linuxbrew/.linuxbrew/bin/brew)
PREFIX    := $(shell $(BREW) --prefix onnxruntime 2>/dev/null || pkg-config --variable=prefix onnxruntime 2>/dev/null || echo /usr)
HB_PREFIX := $(shell $(BREW) --prefix 2>/dev/null || echo /home/linuxbrew/.linuxbrew)
HB_CELLAR := $(shell $(BREW) --cellar 2>/dev/null || echo /home/linuxbrew/.linuxbrew/Cellar)
HB_LIB    := $(HB_PREFIX)/lib

export CGO_ENABLED     := 1
export CGO_LDFLAGS     := -L$(PREFIX)/lib -L$(HB_LIB) -Wl,-rpath=$(HB_LIB) -Wl,-rpath=$(PREFIX)/lib -lonnxruntime
export LD_LIBRARY_PATH := $(HB_LIB):$(PREFIX)/lib:$(LD_LIBRARY_PATH)
export PKG_CONFIG_PATH := $(HB_LIB)/pkgconfig:$(HB_PREFIX)/share/pkgconfig:$(HB_CELLAR)/xorgproto/2025.1/share/pkgconfig:$(HB_CELLAR)/libayatana-appindicator/0.6.0/lib/pkgconfig:$(PKG_CONFIG_PATH)

SECS ?= 5

.PHONY: build download record run clean

build:
	@[ -f "$(PREFIX)/lib/libonnxruntime.so" ] || { \
		echo "ERROR: libonnxruntime.so not found at $(PREFIX)/lib" >&2; \
		echo "       install it: brew install onnxruntime" >&2; \
		exit 1; \
	}
	go build -buildvcs=false -o gostt ./cmd/gostt
	@echo "==> gostt built: $(PWD)/gostt"

download: build
	./gostt download

record: build
	./gostt record $(SECS)

run: build download
	./gostt record $(SECS)

clean:
	rm -f gostt

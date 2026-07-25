#!/usr/bin/env bash
# scripts/dev-shell.sh
#
# One-line wrapper for entering the gostt dev distrobox with all the
# recommended flags already set (audio passthrough, ONNX runtime libs,
# GOPATH bind-mount).
#
# Usage:
#   ./scripts/dev-shell.sh                      # enter the box
#   ./scripts/dev-shell.sh exec ls -la /tmp    # run a command inside
#   ./scripts/dev-shell.sh exec grep "fx bar" file.txt   # quoted args OK

set -euo pipefail

NAME="${GOSTT_BOX_NAME:-gostt-dev}"
ORT_VER="${ORT_VER:-1.18.0}"
ORT_DIR="/opt/onnxruntime-linux-x64-${ORT_VER}"

EXTRA_FLAGS=(
  --device=/dev/snd
  --device=/dev/uinput
  --group-add=audio
  --group-add=input
)

run_in_box() {
    # Forward the host LD_LIBRARY_PATH into the container so ONNX runtime
    # is reachable for any go-built binary.
    local -a env_args=(
      --env="CGO_ENABLED=1"
      --env="GOSTT_ORT_LIB=${ORT_DIR}/lib/libonnxruntime.so"
      --env="LD_LIBRARY_PATH=${ORT_DIR}/lib:${LD_LIBRARY_PATH:-}"
    )

    if [[ $# -lt 1 ]]; then
        exec distrobox enter "${NAME}" -- "${EXTRA_FLAGS[@]}" "${env_args[@]}"
    fi

    # Build a SINGLE correctly-quoted command string, then hand it to
    # `bash -lc` as one argument. The naive `${*@Q}` emission produces
    # one quoted token per arg, which `bash -lc` mistakenly interprets
    # as positional args ($0, $1, …) instead of one shell snippet.
    # `printf '%q'` emits each arg shell-escaped AND joined with
    # spaces in a single value that's safe to feed to `bash -lc` again.
    local cmd
    printf -v cmd '%q ' "$@"
    cmd="${cmd% }"  # strip trailing space
    exec distrobox enter "${NAME}" -- \
        "${EXTRA_FLAGS[@]}" "${env_args[@]}" \
        bash -lc "$cmd"
}

case "${1:-}" in
    exec) shift; run_in_box "$@" ;;
    *)    run_in_box ;;
esac

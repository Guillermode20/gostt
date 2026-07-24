#!/usr/bin/env bash
# scripts/dev-shell.sh
#
# One-line wrapper for entering the gott dev distrobox with all the
# recommended flags already set (audio passthrough, ONNX runtime libs,
# GOPATH bind-mount).
#
# Usage:
#   ./scripts/dev-shell.sh                      # enter the box
#   ./scripts/dev-shell.sh exec ls -la /tmp    # run a command inside
#   ./scripts/dev-shell.sh exec grep "fx bar" file.txt   # quoted args OK

set -euo pipefail

NAME="${GOTT_BOX_NAME:-gott-dev}"
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
      --env="GOTT_ORT_LIB=${ORT_DIR}/lib/libonnxruntime.so"
      --env="LD_LIBRARY_PATH=${ORT_DIR}/lib:${LD_LIBRARY_PATH:-}"
    )

    if [[ $# -lt 1 ]]; then
        exec distrobox enter "${NAME}" -- "${EXTRA_FLAGS[@]}" "${env_args[@]}"
    fi

    # Build the command string WITHOUT losing per-arg quoting. bash-5+
    # supports the `${array[*]@Q}` operator that emits each element
    # quoted exactly as it would appear at command-parse time. Fall back
    # to a manual escape on older bash.
    local cmd
    if ((BASH_VERSINFO[0] >= 5)); then
        cmd="${*@Q}"
    else
        # Manual escape for bash 4.x.
        local esc=()
        local a
        for a in "$@"; do
            esc+=("$(printf '%s' "$a" | sed "s/'/'\\\\''/g; 1s/^/'/; \$s/\$/'/")")
        done
        cmd="${esc[*]}"
    fi
    exec distrobox enter "${NAME}" -- \
        "${EXTRA_FLAGS[@]}" "${env_args[@]}" \
        bash -lc "$cmd"
}

case "${1:-}" in
    exec) shift; run_in_box "$@" ;;
    *)    run_in_box ;;
esac

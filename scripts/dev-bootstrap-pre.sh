#!/usr/bin/env bash
# scripts/dev-bootstrap-pre.sh — runs BEFORE the distrobox image is
# pulled. Currently a no-op but kept because distrobox.ini references
# it; future hooks (e.g. apt-update marker file generation) live here.
echo "==> dev-bootstrap-pre: nothing to do before pull"

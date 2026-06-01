#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/gatsbyli/applsci-volatility-wavelet-ml"
MANUSCRIPT_DIR="$ROOT/paper/manuscript"
TECTONIC_BIN="$ROOT/tools/tectonic/tectonic"

cd "$MANUSCRIPT_DIR"
"$TECTONIC_BIN" main.tex --keep-logs --keep-intermediates
echo "Compiled manuscript to: $MANUSCRIPT_DIR/main.pdf"

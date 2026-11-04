#!/usr/bin/env bash
# Record a ground-truth "real" sample from the laptop mic.
# Usage: scripts/record_real.sh [seconds] [name]
set -euo pipefail
SECS="${1:-8}"
NAME="${2:-real_$(date +%s)}"
cd "$(dirname "$0")/.."
mkdir -p data/real
echo "Recording ${SECS}s to data/real/${NAME}.wav ..."
arecord -d "$SECS" -f S16_LE -r 16000 -c 1 "data/real/${NAME}.wav"
echo "Done. Analyze it in the app's file tab."

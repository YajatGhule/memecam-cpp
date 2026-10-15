#!/usr/bin/env bash
# Pull the .tflite models out of MediaPipe's .task bundles (zip files, the hand one nested).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-$HOME/Projects/meme-cam/models}"
OUT="$ROOT/models"; TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT"
for t in gesture_recognizer face_landmarker pose_landmarker_lite; do
  mkdir -p "$TMP/$t" && unzip -qo "$SRC/$t.task" -d "$TMP/$t"
done
for inner in "$TMP"/gesture_recognizer/*.task; do unzip -qo "$inner" -d "$TMP/gesture_recognizer"; done
find "$TMP" -name '*.tflite' -exec cp {} "$OUT/" \;
ls -la "$OUT"

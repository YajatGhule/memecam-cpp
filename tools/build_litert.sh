#!/usr/bin/env bash
# Build LiteRT (TensorFlow Lite C++ with XNNPACK) once into third_party/litert-install.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TF_TAG="${TF_TAG:-v2.21.0}"
SRC="$ROOT/third_party/tensorflow"
BUILD="$ROOT/third_party/litert-build"
[ -d "$SRC" ] || git clone --depth 1 --branch "$TF_TAG" https://github.com/tensorflow/tensorflow.git "$SRC"
cmake -S "$SRC/tensorflow/lite" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DTFLITE_ENABLE_XNNPACK=ON -DTFLITE_ENABLE_GPU=OFF -DTFLITE_ENABLE_INSTALL=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CXX_FLAGS="-Wno-error -include cstdint" -DCMAKE_C_FLAGS="-Wno-error"
cmake --build "$BUILD" -j"$(nproc)" --target tensorflow-lite
echo "LiteRT built in $BUILD"

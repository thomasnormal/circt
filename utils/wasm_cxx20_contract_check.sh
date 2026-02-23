#!/usr/bin/env bash
set -euo pipefail

CONFIG_SCRIPT="${1:-utils/configure_wasm_build.sh}"
FIRRTL_FILE="${2:-tools/circt-tblgen/FIRRTLAnnotationsGen.cpp}"

if [[ ! -x "$CONFIG_SCRIPT" ]]; then
  echo "[wasm-cxx20-contract] missing executable: $CONFIG_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$FIRRTL_FILE" ]]; then
  echo "[wasm-cxx20-contract] missing source file: $FIRRTL_FILE" >&2
  exit 1
fi

configure_cmd="$(
  BUILD_DIR=/tmp/wasm-cxx20-out LLVM_SRC_DIR=/tmp/wasm-cxx20-llvm \
    "$CONFIG_SCRIPT" --print-cmake-command
)"

if ! grep -Fq -- "-DCMAKE_CXX_STANDARD=20" <<<"$configure_cmd"; then
  echo "[wasm-cxx20-contract] configure command missing -DCMAKE_CXX_STANDARD=20" >&2
  exit 1
fi

if ! grep -Fq -- "ArrayRef<Parameter> getFields() const;" "$FIRRTL_FILE"; then
  echo "[wasm-cxx20-contract] ObjectType::getFields declaration is not out-of-line" >&2
  exit 1
fi

if ! grep -Fq -- "ArrayRef<Parameter> ObjectType::getFields() const { return fields; }" "$FIRRTL_FILE"; then
  echo "[wasm-cxx20-contract] missing out-of-line ObjectType::getFields definition" >&2
  exit 1
fi

tmp_err="$(mktemp)"
trap 'rm -f "$tmp_err"' EXIT
if CMAKE_CXX_STANDARD=17 "$CONFIG_SCRIPT" --print-cmake-command > /dev/null 2>"$tmp_err"; then
  echo "[wasm-cxx20-contract] configure script accepted unsupported C++ standard override (17)" >&2
  exit 1
fi
if ! grep -Fq -- "CMAKE_CXX_STANDARD must be >= 20" "$tmp_err"; then
  echo "[wasm-cxx20-contract] missing explicit floor diagnostic for C++20 requirement" >&2
  exit 1
fi

echo "[wasm-cxx20-contract] PASS"

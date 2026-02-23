#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-wasm}"
LLVM_SRC_DIR="${LLVM_SRC_DIR:-$ROOT_DIR/llvm/llvm}"
EMCMAKE_BIN="${EMCMAKE_BIN:-emcmake}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
CMAKE_CXX_STANDARD="${CMAKE_CXX_STANDARD:-20}"
LLVM_ENABLE_ASSERTIONS="${LLVM_ENABLE_ASSERTIONS:-ON}"
BUILD_SHARED_LIBS="${BUILD_SHARED_LIBS:-OFF}"

if [[ ! "$CMAKE_CXX_STANDARD" =~ ^[0-9]+$ ]]; then
  echo "[wasm-configure] CMAKE_CXX_STANDARD must be a numeric integer (got $CMAKE_CXX_STANDARD)" >&2
  exit 1
fi

if (( CMAKE_CXX_STANDARD < 20 )); then
  echo "[wasm-configure] CMAKE_CXX_STANDARD must be >= 20 (got $CMAKE_CXX_STANDARD)" >&2
  exit 1
fi

print_only=0
extra_cmake_args=()
while (($#)); do
  case "$1" in
    --print-cmake-command)
      print_only=1
      shift
      ;;
    --)
      shift
      while (($#)); do
        extra_cmake_args+=("$1")
        shift
      done
      ;;
    *)
      extra_cmake_args+=("$1")
      shift
      ;;
  esac
done

cmake_args=(
  -G Ninja
  -S "$LLVM_SRC_DIR"
  -B "$BUILD_DIR"
  -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS"
  -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE"
  -DCMAKE_CXX_STANDARD="$CMAKE_CXX_STANDARD"
  -DLLVM_ENABLE_ASSERTIONS="$LLVM_ENABLE_ASSERTIONS"
  -DLLVM_ENABLE_PROJECTS=mlir
  -DLLVM_EXTERNAL_PROJECTS=circt
  -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR="$ROOT_DIR"
  -DLLVM_TARGETS_TO_BUILD=WebAssembly
  -DMLIR_ENABLE_EXECUTION_ENGINE=OFF
  -DLLVM_ENABLE_THREADS=OFF
  -DCIRCT_SLANG_FRONTEND_ENABLED=ON
)

if [[ "$print_only" -eq 1 ]]; then
  printf "%q " "$EMCMAKE_BIN" "$CMAKE_BIN" "${cmake_args[@]}" "${extra_cmake_args[@]}"
  printf "\n"
  exit 0
fi

if ! command -v "$EMCMAKE_BIN" >/dev/null 2>&1; then
  echo "[wasm-configure] missing emcmake wrapper: $EMCMAKE_BIN" >&2
  echo "  install/source emsdk and ensure emcmake is on PATH" >&2
  exit 1
fi

if ! command -v "$CMAKE_BIN" >/dev/null 2>&1; then
  echo "[wasm-configure] missing cmake binary: $CMAKE_BIN" >&2
  exit 1
fi

"$EMCMAKE_BIN" "$CMAKE_BIN" "${cmake_args[@]}" "${extra_cmake_args[@]}"

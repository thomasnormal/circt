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
CIRCT_SIM_WASM_ENABLE_NODERAWFS="${CIRCT_SIM_WASM_ENABLE_NODERAWFS:-ON}"
CIRCT_SIM_WASM_ALLOW_MEMORY_GROWTH="${CIRCT_SIM_WASM_ALLOW_MEMORY_GROWTH:-ON}"
CIRCT_SIM_WASM_STACK_SIZE="${CIRCT_SIM_WASM_STACK_SIZE:-33554432}"
CIRCT_VERILOG_WASM_ENABLE_NODERAWFS="${CIRCT_VERILOG_WASM_ENABLE_NODERAWFS:-ON}"
CIRCT_VERILOG_WASM_ALLOW_MEMORY_GROWTH="${CIRCT_VERILOG_WASM_ALLOW_MEMORY_GROWTH:-ON}"
CIRCT_VERILOG_WASM_STACK_SIZE="${CIRCT_VERILOG_WASM_STACK_SIZE:-33554432}"

validate_on_off_env() {
  local name="$1"
  local value="$2"
  if [[ "$value" != "ON" && "$value" != "OFF" ]]; then
    echo "[wasm-configure] $name must be ON or OFF (got $value)" >&2
    exit 1
  fi
}

if [[ ! "$CMAKE_CXX_STANDARD" =~ ^[0-9]+$ ]]; then
  echo "[wasm-configure] CMAKE_CXX_STANDARD must be a numeric integer (got $CMAKE_CXX_STANDARD)" >&2
  exit 1
fi

if (( CMAKE_CXX_STANDARD < 20 )); then
  echo "[wasm-configure] CMAKE_CXX_STANDARD must be >= 20 (got $CMAKE_CXX_STANDARD)" >&2
  exit 1
fi

validate_on_off_env "LLVM_ENABLE_ASSERTIONS" "$LLVM_ENABLE_ASSERTIONS"
validate_on_off_env "BUILD_SHARED_LIBS" "$BUILD_SHARED_LIBS"
validate_on_off_env "CIRCT_SIM_WASM_ENABLE_NODERAWFS" "$CIRCT_SIM_WASM_ENABLE_NODERAWFS"
validate_on_off_env "CIRCT_SIM_WASM_ALLOW_MEMORY_GROWTH" "$CIRCT_SIM_WASM_ALLOW_MEMORY_GROWTH"
validate_on_off_env "CIRCT_VERILOG_WASM_ENABLE_NODERAWFS" "$CIRCT_VERILOG_WASM_ENABLE_NODERAWFS"
validate_on_off_env "CIRCT_VERILOG_WASM_ALLOW_MEMORY_GROWTH" "$CIRCT_VERILOG_WASM_ALLOW_MEMORY_GROWTH"

if [[ ! "$CIRCT_SIM_WASM_STACK_SIZE" =~ ^[0-9]+$ ]]; then
  echo "[wasm-configure] CIRCT_SIM_WASM_STACK_SIZE must be a numeric integer (got $CIRCT_SIM_WASM_STACK_SIZE)" >&2
  exit 1
fi

if (( CIRCT_SIM_WASM_STACK_SIZE < 1 )); then
  echo "[wasm-configure] CIRCT_SIM_WASM_STACK_SIZE must be >= 1 (got $CIRCT_SIM_WASM_STACK_SIZE)" >&2
  exit 1
fi

if [[ ! "$CIRCT_VERILOG_WASM_STACK_SIZE" =~ ^[0-9]+$ ]]; then
  echo "[wasm-configure] CIRCT_VERILOG_WASM_STACK_SIZE must be a numeric integer (got $CIRCT_VERILOG_WASM_STACK_SIZE)" >&2
  exit 1
fi

if (( CIRCT_VERILOG_WASM_STACK_SIZE < 1 )); then
  echo "[wasm-configure] CIRCT_VERILOG_WASM_STACK_SIZE must be >= 1 (got $CIRCT_VERILOG_WASM_STACK_SIZE)" >&2
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
  -DCROSS_TOOLCHAIN_FLAGS_NATIVE="-DLLVM_ENABLE_PROJECTS=mlir"
  -DCIRCT_SLANG_FRONTEND_ENABLED=ON
  -DCIRCT_SIM_WASM_ENABLE_NODERAWFS="$CIRCT_SIM_WASM_ENABLE_NODERAWFS"
  -DCIRCT_SIM_WASM_ALLOW_MEMORY_GROWTH="$CIRCT_SIM_WASM_ALLOW_MEMORY_GROWTH"
  -DCIRCT_SIM_WASM_STACK_SIZE="$CIRCT_SIM_WASM_STACK_SIZE"
  -DCIRCT_VERILOG_WASM_ENABLE_NODERAWFS="$CIRCT_VERILOG_WASM_ENABLE_NODERAWFS"
  -DCIRCT_VERILOG_WASM_ALLOW_MEMORY_GROWTH="$CIRCT_VERILOG_WASM_ALLOW_MEMORY_GROWTH"
  -DCIRCT_VERILOG_WASM_STACK_SIZE="$CIRCT_VERILOG_WASM_STACK_SIZE"
)

if [[ "$print_only" -eq 1 ]]; then
  printf "%q " "$EMCMAKE_BIN" "$CMAKE_BIN" "${cmake_args[@]}"
  if ((${#extra_cmake_args[@]})); then
    printf "%q " "${extra_cmake_args[@]}"
  fi
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

if [[ ! -d "$LLVM_SRC_DIR" ]]; then
  echo "[wasm-configure] missing LLVM source directory: $LLVM_SRC_DIR" >&2
  exit 1
fi

if ((${#extra_cmake_args[@]})); then
  "$EMCMAKE_BIN" "$CMAKE_BIN" "${cmake_args[@]}" "${extra_cmake_args[@]}"
else
  "$EMCMAKE_BIN" "$CMAKE_BIN" "${cmake_args[@]}"
fi

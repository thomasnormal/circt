#!/usr/bin/env bash
set -euo pipefail

SCRIPT="${1:-utils/configure_wasm_build.sh}"

if [[ ! -x "$SCRIPT" ]]; then
  echo "[wasm-config-contract] missing executable: $SCRIPT" >&2
  exit 1
fi

output="$(BUILD_DIR=/tmp/wasm-out LLVM_SRC_DIR=/tmp/llvm-src "$SCRIPT" --print-cmake-command)"

required_cmd_tokens=(
  "emcmake"
  "-S /tmp/llvm-src"
  "-B /tmp/wasm-out"
  "-DLLVM_TARGETS_TO_BUILD=WebAssembly"
  "-DMLIR_ENABLE_EXECUTION_ENGINE=OFF"
  "-DLLVM_ENABLE_THREADS=OFF"
  "-DCIRCT_SLANG_FRONTEND_ENABLED=ON"
  "-DCIRCT_SIM_WASM_ENABLE_NODERAWFS="
)

for token in "${required_cmd_tokens[@]}"; do
  if ! grep -Fq -- "$token" <<<"$output"; then
    echo "[wasm-config-contract] missing token in configure command: $token" >&2
    exit 1
  fi
done

required_source_tokens=(
  'command -v "$EMCMAKE_BIN"'
  'command -v "$CMAKE_BIN"'
  "validate_on_off_env"
  'validate_on_off_env "LLVM_ENABLE_ASSERTIONS" "$LLVM_ENABLE_ASSERTIONS"'
  'validate_on_off_env "BUILD_SHARED_LIBS" "$BUILD_SHARED_LIBS"'
  'validate_on_off_env "CIRCT_SIM_WASM_ENABLE_NODERAWFS" "$CIRCT_SIM_WASM_ENABLE_NODERAWFS"'
  "must be ON or OFF"
)

for token in "${required_source_tokens[@]}"; do
  if ! grep -Fq -- "$token" "$SCRIPT"; then
    echo "[wasm-config-contract] missing source check in configure script: $token" >&2
    exit 1
  fi
done

tmp_err="$(mktemp)"
trap 'rm -f "$tmp_err"' EXIT
if CIRCT_SIM_WASM_ENABLE_NODERAWFS=maybe "$SCRIPT" --print-cmake-command > /dev/null 2>"$tmp_err"; then
  echo "[wasm-config-contract] configure script accepted invalid CIRCT_SIM_WASM_ENABLE_NODERAWFS override" >&2
  exit 1
fi
if ! grep -Fq -- "CIRCT_SIM_WASM_ENABLE_NODERAWFS must be ON or OFF" "$tmp_err"; then
  echo "[wasm-config-contract] missing explicit ON/OFF diagnostic for CIRCT_SIM_WASM_ENABLE_NODERAWFS" >&2
  exit 1
fi

if LLVM_ENABLE_ASSERTIONS=enabled "$SCRIPT" --print-cmake-command > /dev/null 2>"$tmp_err"; then
  echo "[wasm-config-contract] configure script accepted invalid LLVM_ENABLE_ASSERTIONS override" >&2
  exit 1
fi
if ! grep -Fq -- "LLVM_ENABLE_ASSERTIONS must be ON or OFF" "$tmp_err"; then
  echo "[wasm-config-contract] missing explicit ON/OFF diagnostic for LLVM_ENABLE_ASSERTIONS" >&2
  exit 1
fi

if BUILD_SHARED_LIBS=static "$SCRIPT" --print-cmake-command > /dev/null 2>"$tmp_err"; then
  echo "[wasm-config-contract] configure script accepted invalid BUILD_SHARED_LIBS override" >&2
  exit 1
fi
if ! grep -Fq -- "BUILD_SHARED_LIBS must be ON or OFF" "$tmp_err"; then
  echo "[wasm-config-contract] missing explicit ON/OFF diagnostic for BUILD_SHARED_LIBS" >&2
  exit 1
fi

echo "[wasm-config-contract] PASS"

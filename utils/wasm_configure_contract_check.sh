#!/usr/bin/env bash
set -euo pipefail

SCRIPT="${1:-utils/configure_wasm_build.sh}"

if [[ ! -x "$SCRIPT" ]]; then
  echo "[wasm-config-contract] missing executable: $SCRIPT" >&2
  exit 1
fi

output="$(BUILD_DIR=/tmp/wasm-out LLVM_SRC_DIR=/tmp/llvm-src "$SCRIPT" --print-cmake-command)"

required_tokens=(
  "emcmake"
  "-S /tmp/llvm-src"
  "-B /tmp/wasm-out"
  "-DLLVM_TARGETS_TO_BUILD=WebAssembly"
  "-DMLIR_ENABLE_EXECUTION_ENGINE=OFF"
  "-DLLVM_ENABLE_THREADS=OFF"
  "-DCIRCT_SLANG_FRONTEND_ENABLED=ON"
)

for token in "${required_tokens[@]}"; do
  if ! grep -Fq -- "$token" <<<"$output"; then
    echo "[wasm-config-contract] missing token in configure command: $token" >&2
    exit 1
  fi
done

echo "[wasm-config-contract] PASS"

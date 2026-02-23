#!/usr/bin/env bash
set -euo pipefail

PLUSARGS_SCRIPT="${1:-utils/wasm_plusargs_reentry_check.sh}"
RG_SCRIPT="${2:-utils/wasm_resource_guard_default_check.sh}"

for script in "$PLUSARGS_SCRIPT" "$RG_SCRIPT"; do
  if [[ ! -x "$script" ]]; then
    echo "[wasm-runtime-helpers-contract] missing executable: $script" >&2
    exit 1
  fi
done

plusargs_tokens=(
  'command -v "$NODE_BIN"'
  "[wasm-plusargs] missing Node.js runtime"
)

for token in "${plusargs_tokens[@]}"; do
  if ! grep -Fq -- "$token" "$PLUSARGS_SCRIPT"; then
    echo "[wasm-runtime-helpers-contract] missing token in plusargs helper: $token" >&2
    exit 1
  fi
done

rg_tokens=(
  'command -v "$NODE_BIN"'
  "[wasm-rg-default] missing Node.js runtime"
  'BMC_TEST_INPUT="${BMC_TEST_INPUT:-'
  'SIM_TEST_INPUT="${SIM_TEST_INPUT:-'
  'SV_TEST_INPUT="${SV_TEST_INPUT:-'
  'if [[ -f "$VERILOG_JS" ]]; then'
  'if [[ ! -f "$SV_TEST_INPUT" ]]; then'
  "[wasm-rg-default] missing test input:"
)

for token in "${rg_tokens[@]}"; do
  if ! grep -Fq -- "$token" "$RG_SCRIPT"; then
    echo "[wasm-runtime-helpers-contract] missing token in default-guard helper: $token" >&2
    exit 1
  fi
done

echo "[wasm-runtime-helpers-contract] PASS"

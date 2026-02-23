#!/usr/bin/env bash
set -euo pipefail

SMOKE_SCRIPT="${1:-utils/run_wasm_smoke.sh}"

if [[ ! -f "$SMOKE_SCRIPT" ]]; then
  echo "[wasm-smoke-contract] missing script: $SMOKE_SCRIPT" >&2
  exit 1
fi

required_tokens=(
  "WASM_SKIP_BUILD"
  "WASM_REQUIRE_CLEAN_CROSSCOMPILE"
  "circt-bmc.wasm"
  "circt-sim.wasm"
  "Functional: circt-verilog stdin (.sv) -> IR"
  "Functional: circt-verilog (.sv) -> circt-sim"
  "Re-entry: circt-verilog callMain help -> run"
  "Re-entry: circt-verilog run -> run"
)

for token in "${required_tokens[@]}"; do
  if ! grep -Fq -- "$token" "$SMOKE_SCRIPT"; then
    echo "[wasm-smoke-contract] missing token in smoke script: $token" >&2
    exit 1
  fi
done

echo "[wasm-smoke-contract] PASS"

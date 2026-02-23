#!/usr/bin/env bash
set -euo pipefail

SMOKE_SCRIPT="${1:-utils/run_wasm_smoke.sh}"

if [[ ! -f "$SMOKE_SCRIPT" ]]; then
  echo "[wasm-smoke-contract] missing script: $SMOKE_SCRIPT" >&2
  exit 1
fi

required_tokens=(
  "WASM_SKIP_BUILD"
  "WASM_CHECK_CXX20_WARNINGS"
  "WASM_REQUIRE_CLEAN_CROSSCOMPILE"
  "WASM_REQUIRE_VERILOG"
  "validate_bool_env"
  'validate_bool_env "WASM_REQUIRE_VERILOG" "$WASM_REQUIRE_VERILOG"'
  'validate_bool_env "WASM_SKIP_BUILD" "$WASM_SKIP_BUILD"'
  'validate_bool_env "WASM_REQUIRE_CLEAN_CROSSCOMPILE" "$WASM_REQUIRE_CLEAN_CROSSCOMPILE"'
  'invalid $name value'
  "invalid WASM_CHECK_CXX20_WARNINGS"
  'command -v "$NODE_BIN"'
  'command -v ninja'
  "utils/wasm_cxx20_warning_check.sh"
  "git -C llvm diff --quiet -- llvm/cmake/modules/CrossCompile.cmake"
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

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
  "VCD_PATH"
  'expected VCD output to include \$enddefinitions'
  'expected SV pipeline VCD to include \$enddefinitions'
  "validate_bool_env"
  "validate_positive_int_env"
  'validate_positive_int_env "NINJA_JOBS" "$NINJA_JOBS"'
  'validate_bool_env "WASM_REQUIRE_VERILOG" "$WASM_REQUIRE_VERILOG"'
  'validate_bool_env "WASM_SKIP_BUILD" "$WASM_SKIP_BUILD"'
  'validate_bool_env "WASM_REQUIRE_CLEAN_CROSSCOMPILE" "$WASM_REQUIRE_CLEAN_CROSSCOMPILE"'
  'invalid $name value'
  "invalid WASM_CHECK_CXX20_WARNINGS"
  "invalid VCD_PATH value: empty path"
  'command -v "$NODE_BIN"'
  'command -v ninja'
  "utils/wasm_cxx20_warning_check.sh"
  "REENTRY_HELPER"
  "PLUSARGS_HELPER"
  "RESOURCE_GUARD_HELPER"
  'REENTRY_VCD="/tmp/reentry-${BASHPID}.vcd"'
  'REENTRY_RUN1_VCD="/tmp/reentry-run1-${BASHPID}.vcd"'
  'REENTRY_RUN2_VCD="/tmp/reentry-run2-${BASHPID}.vcd"'
  '--vcd "$REENTRY_VCD"'
  '--first --resource-guard=false --vcd "$REENTRY_RUN1_VCD"'
  '--second --resource-guard=false --vcd "$REENTRY_RUN2_VCD"'
  'missing helper script: $REENTRY_HELPER'
  'missing executable helper script: $PLUSARGS_HELPER'
  'missing executable helper script: $RESOURCE_GUARD_HELPER'
  "git -C llvm diff --quiet -- llvm/cmake/modules/CrossCompile.cmake"
  "unable to inspect llvm submodule CrossCompile.cmake status"
  'git_rc=$?'
  'if [[ "$git_rc" -eq 1 ]]; then'
  'ninja target query failed; inferring circt-verilog support from existing artifacts'
  'ninja target query failed and circt-verilog is optional; skipping SV frontend checks'
  'elif [[ "$WASM_SKIP_BUILD" == "1" && "$WASM_REQUIRE_VERILOG" != "1" ]]; then'
  'failed to query ninja targets'
  'WASM_SKIP_BUILD" == "1" && -s "$VERILOG_JS" && -s "$VERILOG_WASM"'
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

forbidden_tokens=(
  'expected VCD output to declare at least one \$var'
  'expected SV pipeline VCD to declare at least one \$var'
)

for token in "${forbidden_tokens[@]}"; do
  if grep -Fq -- "$token" "$SMOKE_SCRIPT"; then
    echo "[wasm-smoke-contract] stale token still present in smoke script: $token" >&2
    exit 1
  fi
done

echo "[wasm-smoke-contract] PASS"

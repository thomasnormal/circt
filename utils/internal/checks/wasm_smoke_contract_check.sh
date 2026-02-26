#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/common.sh"
repo_root="$(checks_repo_root "$script_dir")"
cd "$repo_root"

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
  "REENTRY_STDOUT_CAPTURE_HELPER"
  "PLUSARGS_HELPER"
  "RESOURCE_GUARD_HELPER"
  "UVM_STUB_VCD_HELPER"
  "UVM_PKG_MEMFS_HELPER"
  "VPI_STARTUP_YIELD_HELPER"
  "VPI_REENTRY_ISOLATION_HELPER"
  "VERSION_REENTRY_HELPER"
  "VERILOG_ANALYSIS_HELPER"
  'REENTRY_VCD="/tmp/reentry-${SCRIPT_PID}.vcd"'
  'REENTRY_RUN1_VCD="/tmp/reentry-run1-${SCRIPT_PID}.vcd"'
  'REENTRY_RUN2_VCD="/tmp/reentry-run2-${SCRIPT_PID}.vcd"'
  '--vcd "$REENTRY_VCD"'
  '--first --resource-guard=false --vcd "$REENTRY_RUN1_VCD"'
  '--second --resource-guard=false --vcd "$REENTRY_RUN2_VCD"'
  'missing helper script: $REENTRY_HELPER'
  'missing executable helper script: $REENTRY_STDOUT_CAPTURE_HELPER'
  'missing executable helper script: $PLUSARGS_HELPER'
  'missing executable helper script: $RESOURCE_GUARD_HELPER'
  'missing executable helper script: $UVM_STUB_VCD_HELPER'
  'missing executable helper script: $UVM_PKG_MEMFS_HELPER'
  'missing executable helper script: $VPI_STARTUP_YIELD_HELPER'
  'missing executable helper script: $VPI_REENTRY_ISOLATION_HELPER'
  'missing executable helper script: $VERSION_REENTRY_HELPER'
  'missing executable helper script: $VERILOG_ANALYSIS_HELPER'
  "UVM stub frontend+sim+VCD"
  "UVM pkg frontend MEMFS re-entry"
  "utils/wasm_uvm_stub_vcd_check.sh"
  "utils/wasm_uvm_pkg_memfs_reentry_check.sh"
  "utils/wasm_callmain_reentry_stdout_capture_check.sh"
  "utils/wasm_vpi_startup_yield_check.sh"
  "utils/wasm_vpi_reentry_callback_isolation_check.sh"
  "utils/wasm_version_reentry_check.sh"
  "utils/wasm_verilog_analysis_fallback_check.sh"
  "VPI startup-yield hook (cb_rtn=0)"
  "VPI callback re-entry isolation"
  "wasm verilog semantic-analysis fallback checks"
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
  "--no-uvm-auto-include --ir-llhd --single-unit --format=sv"
  "Re-entry: circt-verilog callMain help -> run"
  "Re-entry: callMain stdout capture"
  "Re-entry: --version banner duplication"
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

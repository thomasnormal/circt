#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/common.sh"
repo_root="$(checks_repo_root "$script_dir")"
cd "$repo_root"

WORKFLOW="${1:-.github/workflows/wasmSmoke.yml}"

if [[ ! -f "$WORKFLOW" ]]; then
  echo "[wasm-ci-contract] missing workflow: $WORKFLOW" >&2
  exit 1
fi

required_tokens=(
  "utils/configure_wasm_build.sh"
  "utils/internal/checks/wasm_configure_behavior_check.sh"
  "utils/internal/checks/wasm_configure_behavior_contract_check.sh"
  "utils/internal/checks/wasm_configure_contract_check.sh"
  "utils/internal/checks/wasm_cxx20_contract_check.sh"
  "utils/wasm_cxx20_warning_check.sh"
  "utils/internal/checks/wasm_cxx20_warning_behavior_check.sh"
  "utils/internal/checks/wasm_cxx20_warning_contract_check.sh"
  "utils/internal/checks/wasm_runtime_helpers_behavior_check.sh"
  "utils/internal/checks/wasm_runtime_helpers_behavior_contract_check.sh"
  "utils/internal/checks/wasm_runtime_helpers_contract_check.sh"
  "utils/internal/checks/wasm_smoke_contract_check.sh"
  "utils/internal/checks/wasm_ci_contract_check.sh"
  "utils/repro_wasm_uvm_browser_assert.mjs --expect-pass"
  "utils/run_wasm_smoke.sh"
  "WASM_REQUIRE_VERILOG=1"
  "WASM_SKIP_BUILD=1"
  "WASM_CHECK_CXX20_WARNINGS=1"
  "WASM_CHECK_CXX20_WARNINGS=0"
  "WASM_REQUIRE_CLEAN_CROSSCOMPILE=1"
)

for token in "${required_tokens[@]}"; do
  if ! grep -Fq -- "$token" "$WORKFLOW"; then
    echo "[wasm-ci-contract] missing token in workflow: $token" >&2
    exit 1
  fi
done

echo "[wasm-ci-contract] PASS"

#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/common.sh"
repo_root="$(checks_repo_root "$script_dir")"
cd "$repo_root"

yosys_sva_dir="${YOSYS_SVA_DIR:-/home/thomas-ahle/yosys/tests/sva}"
if [[ ! -d "$yosys_sva_dir" ]]; then
  echo "SKIP: yosys SVA dir not found: $yosys_sva_dir"
  exit 0
fi

log="$(mktemp)"
trap 'rm -f "$log"' EXIT

set +e
CIRCT_SIM=/bin/false \
BMC_SMOKE_ONLY=1 \
TEST_FILTER='^sva_value_change_sim$' \
utils/run_yosys_sva_circt_bmc.sh "$yosys_sva_dir" >"$log" 2>&1
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "expected rc=0 for sim-only smoke case" >&2
  echo "--- log ---" >&2
  cat "$log" >&2
  exit 1
fi

skip_count="$(rg -c '^SKIP\(sim-only\): sva_value_change_sim$' "$log")"
if [[ "$skip_count" -ne 2 ]]; then
  echo "expected two sim-only skips (pass+fail), got $skip_count" >&2
  echo "--- log ---" >&2
  cat "$log" >&2
  exit 1
fi

if rg -q '^FAIL\(pass\): sva_value_change_sim$' "$log"; then
  echo "unexpected FAIL(pass) for sim-only smoke case" >&2
  echo "--- log ---" >&2
  cat "$log" >&2
  exit 1
fi

echo "PASS: yosys BMC smoke skips sim-only case for both modes"

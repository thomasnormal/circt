#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/common.sh"
repo_root="$(checks_repo_root "$script_dir")"
cd "$repo_root"

sv_tests_dir="${SV_TESTS_DIR:-/home/thomas-ahle/sv-tests}"
if [[ ! -d "$sv_tests_dir/tests" ]]; then
  echo "SKIP: sv-tests not found at $sv_tests_dir"
  exit 0
fi

log="$(mktemp)"
trap 'rm -f "$log"' EXIT

set +e
CIRCT_VERILOG=/nonexistent/circt-verilog \
TEST_FILTER='^16\.2--assert$' \
LEC_SMOKE_ONLY=1 \
utils/run_sv_tests_circt_lec.sh "$sv_tests_dir" >"$log" 2>&1
rc=$?
set -e

if [[ "$rc" -eq 0 ]]; then
  echo "expected non-zero exit when LEC summary contains errors" >&2
  echo "--- log ---" >&2
  cat "$log" >&2
  exit 1
fi

if ! rg -q 'error=1' "$log"; then
  echo "expected summary to report error=1" >&2
  echo "--- log ---" >&2
  cat "$log" >&2
  exit 1
fi

echo "PASS: run_sv_tests_circt_lec exits non-zero when errors are present"

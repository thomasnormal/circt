#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LIT_PY="${LIT_PY:-$REPO_ROOT/llvm/llvm/utils/lit/lit.py}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-test}"
SUITE_DIR="${SUITE_DIR:-$BUILD_DIR/test/Tools/circt-sim}"
FILTER_BASE="${FILTER_BASE:-(timeout-no-spurious-vtable-warning|wasm-plusargs-reentry|uvm-phase-add-duplicate-fast-path)}"
FILTER_OVERRIDE="${FILTER:-}"
RUN_NATIVE_SV_LIT="${RUN_NATIVE_SV_LIT:-0}"
NATIVE_SV_FILTER="${NATIVE_SV_FILTER:-(wasm-uvm-stub-vcd|vpi-string-put-value-test|vpi-string-put-value-delayed-test)}"
RUN_SMOKE="${RUN_SMOKE:-1}"
SMOKE_SCRIPT="${SMOKE_SCRIPT:-$REPO_ROOT/utils/run_wasm_smoke.sh}"
SMOKE_WASM_SKIP_BUILD="${SMOKE_WASM_SKIP_BUILD:-1}"
SMOKE_WASM_CHECK_CXX20_WARNINGS="${SMOKE_WASM_CHECK_CXX20_WARNINGS:-0}"
SMOKE_WASM_REQUIRE_VERILOG="${SMOKE_WASM_REQUIRE_VERILOG:-1}"
SMOKE_WASM_REQUIRE_CLEAN_CROSSCOMPILE="${SMOKE_WASM_REQUIRE_CLEAN_CROSSCOMPILE:-0}"

if [[ ! -f "$LIT_PY" ]]; then
  echo "[wasm-regressions] missing lit runner: $LIT_PY" >&2
  exit 1
fi
if [[ ! -d "$SUITE_DIR" ]]; then
  echo "[wasm-regressions] missing lit suite: $SUITE_DIR" >&2
  exit 1
fi

FILTER="$FILTER_BASE"
if [[ -n "$FILTER_OVERRIDE" ]]; then
  FILTER="$FILTER_OVERRIDE"
elif [[ "$RUN_NATIVE_SV_LIT" == "1" ]]; then
  FILTER="($FILTER_BASE|$NATIVE_SV_FILTER)"
fi

tmp_log="$(mktemp)"
trap 'rm -f "$tmp_log"' EXIT

set +e
python3 "$LIT_PY" -sv --show-xfail "$SUITE_DIR" --filter "$FILTER" 2>&1 | tee "$tmp_log"
lit_rc=${PIPESTATUS[0]}
set -e

failures=0
if grep -q '^FAIL: ' "$tmp_log"; then
  failures="$(grep -c '^FAIL: ' "$tmp_log")"
fi

xfails=0
if grep -q '^XFAIL: ' "$tmp_log"; then
  xfails="$(grep -c '^XFAIL: ' "$tmp_log")"
fi

xpasses=0
if grep -q '^XPASS: ' "$tmp_log"; then
  xpasses="$(grep -c '^XPASS: ' "$tmp_log")"
fi

if [[ "$lit_rc" -ne 0 && "$failures" -eq 0 && "$xfails" -eq 0 && "$xpasses" -eq 0 ]]; then
  # lit infrastructure failure (e.g. empty filter match) without test-level rows.
  failures=1
fi

smoke_failures=0
if [[ "$RUN_SMOKE" == "1" ]]; then
  if [[ ! -x "$SMOKE_SCRIPT" ]]; then
    echo "[wasm-regressions] missing executable smoke script: $SMOKE_SCRIPT" >&2
    smoke_failures=1
  else
    if ! WASM_SKIP_BUILD="$SMOKE_WASM_SKIP_BUILD" \
         WASM_CHECK_CXX20_WARNINGS="$SMOKE_WASM_CHECK_CXX20_WARNINGS" \
         WASM_REQUIRE_VERILOG="$SMOKE_WASM_REQUIRE_VERILOG" \
         WASM_REQUIRE_CLEAN_CROSSCOMPILE="$SMOKE_WASM_REQUIRE_CLEAN_CROSSCOMPILE" \
         "$SMOKE_SCRIPT"; then
      smoke_failures=1
    fi
  fi
fi

failures=$((failures + xpasses + smoke_failures))

echo "[wasm-regressions] summary: failures=$failures xfails=$xfails xpasses=$xpasses smoke_failures=$smoke_failures"

if [[ "$failures" -ne 0 || "$xfails" -ne 0 ]]; then
  exit 1
fi

echo "[wasm-regressions] PASS"

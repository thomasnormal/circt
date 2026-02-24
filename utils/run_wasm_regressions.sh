#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LIT_PY="${LIT_PY:-$REPO_ROOT/llvm/llvm/utils/lit/lit.py}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-test}"
SUITE_DIR="${SUITE_DIR:-$BUILD_DIR/test/Tools/circt-sim}"
FILTER="${FILTER:-(timeout-no-spurious-vtable-warning|wasm-uvm-stub-vcd|wasm-plusargs-reentry|vpi-string-put-value-test|vpi-string-put-value-delayed-test|uvm-phase-add-duplicate-fast-path)}"

if [[ ! -f "$LIT_PY" ]]; then
  echo "[wasm-regressions] missing lit runner: $LIT_PY" >&2
  exit 1
fi
if [[ ! -d "$SUITE_DIR" ]]; then
  echo "[wasm-regressions] missing lit suite: $SUITE_DIR" >&2
  exit 1
fi

tmp_log="$(mktemp)"
trap 'rm -f "$tmp_log"' EXIT

python3 "$LIT_PY" -sv --show-xfail "$SUITE_DIR" --filter "$FILTER" 2>&1 | tee "$tmp_log"

failures=0
if grep -q '^FAIL: ' "$tmp_log"; then
  failures="$(grep -c '^FAIL: ' "$tmp_log")"
fi

xfails=0
if grep -q '^XFAIL: ' "$tmp_log"; then
  xfails="$(grep -c '^XFAIL: ' "$tmp_log")"
fi

echo "[wasm-regressions] summary: failures=$failures xfails=$xfails"

if [[ "$failures" -ne 0 || "$xfails" -ne 0 ]]; then
  exit 1
fi

echo "[wasm-regressions] PASS"

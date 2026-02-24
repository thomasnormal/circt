#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_wasm_regressions.sh"

if [[ ! -x "$RUNNER" ]]; then
  echo "[wasm-regressions-behavior] missing executable runner: $RUNNER" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

echo "[wasm-regressions-behavior] case: empty-filter-must-fail-with-summary"
set +e
FILTER='definitely_no_such_test_12345' RUN_SMOKE=0 "$RUNNER" \
  >"$tmpdir/no-match.out" 2>&1
rc=$?
set -e
if [[ "$rc" -eq 0 ]]; then
  echo "[wasm-regressions-behavior] no-match case unexpectedly passed" >&2
  cat "$tmpdir/no-match.out" >&2
  exit 1
fi
if ! grep -q '\[wasm-regressions\] summary: failures=1 xfails=0' "$tmpdir/no-match.out"; then
  echo "[wasm-regressions-behavior] no-match case missing normalized failure summary" >&2
  cat "$tmpdir/no-match.out" >&2
  exit 1
fi

echo "[wasm-regressions-behavior] case: focused-regressions-pass"
RUN_SMOKE=0 "$RUNNER" >"$tmpdir/focused.out" 2>&1
if ! grep -q '\[wasm-regressions\] summary: failures=0 xfails=0' "$tmpdir/focused.out"; then
  echo "[wasm-regressions-behavior] focused case missing zero-failure summary" >&2
  cat "$tmpdir/focused.out" >&2
  exit 1
fi

echo "[wasm-regressions-behavior] PASS"

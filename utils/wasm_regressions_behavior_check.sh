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

echo "[wasm-regressions-behavior] case: lock-contention-fails-cleanly"
lock_file="$tmpdir/wasm-regressions.lock"
exec 9>"$lock_file"
flock -n 9
set +e
WASM_REGRESSIONS_LOCK_FILE="$lock_file" \
WASM_REGRESSIONS_LOCK_WAIT_SECS=0 \
RUN_SMOKE=0 \
  "$RUNNER" >"$tmpdir/lock.out" 2>&1
rc=$?
set -e
exec 9>&-
if [[ "$rc" -eq 0 ]]; then
  echo "[wasm-regressions-behavior] lock contention case unexpectedly passed" >&2
  cat "$tmpdir/lock.out" >&2
  exit 1
fi
if ! grep -q '\[wasm-regressions\] lock busy:' "$tmpdir/lock.out"; then
  echo "[wasm-regressions-behavior] lock contention case missing lock-busy diagnostic" >&2
  cat "$tmpdir/lock.out" >&2
  exit 1
fi

echo "[wasm-regressions-behavior] PASS"

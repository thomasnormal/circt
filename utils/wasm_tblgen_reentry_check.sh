#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-wasm}"
NODE_BIN="${NODE_BIN:-node}"
TBLGEN_JS="${TBLGEN_JS:-$BUILD_DIR/bin/circt-tblgen.js}"
REENTRY_HELPER="${REENTRY_HELPER:-utils/wasm_callmain_reentry_check.js}"
INPUT_TD="${INPUT_TD:-include/circt/Dialect/HW/HW.td}"

if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "[wasm-tblgen-reentry] missing Node.js runtime: $NODE_BIN" >&2
  exit 1
fi
if [[ ! -f "$TBLGEN_JS" ]]; then
  echo "[wasm-tblgen-reentry] missing tool: $TBLGEN_JS" >&2
  exit 1
fi
if [[ ! -f "$REENTRY_HELPER" ]]; then
  echo "[wasm-tblgen-reentry] missing helper: $REENTRY_HELPER" >&2
  exit 1
fi
if [[ ! -f "$INPUT_TD" ]]; then
  echo "[wasm-tblgen-reentry] missing input: $INPUT_TD" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
out_td="$tmpdir/tblgen-reentry-out.td"
out_td_run1="$tmpdir/tblgen-reentry-run1.td"
out_td_run2="$tmpdir/tblgen-reentry-run2.td"

echo "[wasm-tblgen-reentry] same-instance callMain: help -> print-records"
set +e
"$NODE_BIN" "$REENTRY_HELPER" \
  "$TBLGEN_JS" \
  --first --help \
  --second -print-records "$INPUT_TD" -I include -I llvm/include -I llvm/mlir/include -o "$out_td" \
  --expect-substr "OVERVIEW: CIRCT TableGen Generator" \
  --expect-wasm-file-substr "$out_td" "Classes" \
  >"$tmpdir/out.log" 2>"$tmpdir/err.log"
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "[wasm-tblgen-reentry] re-entry run failed" >&2
  cat "$tmpdir/err.log" >&2
  exit 1
fi

echo "[wasm-tblgen-reentry] same-instance callMain: print-records -> print-records"
set +e
"$NODE_BIN" "$REENTRY_HELPER" \
  "$TBLGEN_JS" \
  --first -print-records "$INPUT_TD" -I include -I llvm/include -I llvm/mlir/include -o "$out_td_run1" \
  --second -print-records "$INPUT_TD" -I include -I llvm/include -I llvm/mlir/include -o "$out_td_run2" \
  --expect-wasm-file-substr "$out_td_run1" "Classes" \
  --expect-wasm-file-substr "$out_td_run2" "Classes" \
  >"$tmpdir/runrun.out.log" 2>"$tmpdir/runrun.err.log"
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "[wasm-tblgen-reentry] run->run re-entry failed" >&2
  cat "$tmpdir/runrun.err.log" >&2
  exit 1
fi

echo "[wasm-tblgen-reentry] PASS"

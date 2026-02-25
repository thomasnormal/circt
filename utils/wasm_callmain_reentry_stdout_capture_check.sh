#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-wasm}"
NODE_BIN="${NODE_BIN:-node}"
BMC_JS="${BMC_JS:-$BUILD_DIR/bin/circt-bmc.js}"
REENTRY_HELPER="${REENTRY_HELPER:-utils/wasm_callmain_reentry_check.js}"
INPUT_MLIR="${INPUT_MLIR:-test/Tools/circt-bmc/disable-iff-const-property-unsat.mlir}"

if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "[wasm-reentry-stdout] missing Node.js runtime: $NODE_BIN" >&2
  exit 1
fi
if [[ ! -f "$BMC_JS" ]]; then
  echo "[wasm-reentry-stdout] missing tool: $BMC_JS" >&2
  exit 1
fi
if [[ ! -f "$REENTRY_HELPER" ]]; then
  echo "[wasm-reentry-stdout] missing helper: $REENTRY_HELPER" >&2
  exit 1
fi
if [[ ! -f "$INPUT_MLIR" ]]; then
  echo "[wasm-reentry-stdout] missing input: $INPUT_MLIR" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

echo "[wasm-reentry-stdout] re-entry helper should capture '-o -' stdout text"
set +e
"$NODE_BIN" "$REENTRY_HELPER" \
  "$BMC_JS" \
  --first --resource-guard=false -b 3 --module m_const_prop --emit-smtlib -o - "$INPUT_MLIR" \
  --second --resource-guard=false -b 3 --module m_const_prop --emit-smtlib -o - "$INPUT_MLIR" \
  --expect-substr "(check-sat)" \
  >"$tmpdir/out.log" 2>"$tmpdir/err.log"
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "[wasm-reentry-stdout] helper failed to capture stdout from callMain runs" >&2
  cat "$tmpdir/err.log" >&2
  exit 1
fi
if ! grep -q "\[wasm-reentry\] ok:" "$tmpdir/out.log"; then
  echo "[wasm-reentry-stdout] helper did not report success marker" >&2
  cat "$tmpdir/out.log" >&2
  exit 1
fi

echo "[wasm-reentry-stdout] PASS"

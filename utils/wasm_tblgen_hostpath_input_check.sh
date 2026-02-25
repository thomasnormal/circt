#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-wasm}"
NODE_BIN="${NODE_BIN:-node}"
TBLGEN_JS="${TBLGEN_JS:-$BUILD_DIR/bin/circt-tblgen.js}"
INPUT_TD="${INPUT_TD:-include/circt/Dialect/HW/HW.td}"

if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "[wasm-tblgen-hostpath] missing Node.js runtime: $NODE_BIN" >&2
  exit 1
fi
if [[ ! -f "$TBLGEN_JS" ]]; then
  echo "[wasm-tblgen-hostpath] missing tool: $TBLGEN_JS" >&2
  exit 1
fi
if [[ ! -f "$INPUT_TD" ]]; then
  echo "[wasm-tblgen-hostpath] missing input: $INPUT_TD" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

echo "[wasm-tblgen-hostpath] positional host-path .td input should load in wasm/node mode"
set +e
"$NODE_BIN" "$TBLGEN_JS" \
  -print-records \
  "$INPUT_TD" \
  -I include \
  -I llvm/include \
  -I llvm/mlir/include \
  -o "$tmpdir/records.td" \
  >"$tmpdir/stdout.log" 2>"$tmpdir/stderr.log"
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "[wasm-tblgen-hostpath] run failed (rc=$rc)" >&2
  cat "$tmpdir/stderr.log" >&2
  exit 1
fi
if [[ ! -s "$tmpdir/records.td" ]]; then
  echo "[wasm-tblgen-hostpath] expected output missing: $tmpdir/records.td" >&2
  exit 1
fi
if ! grep -q "Classes" "$tmpdir/records.td"; then
  echo "[wasm-tblgen-hostpath] output missing expected TableGen records banner" >&2
  head -n 40 "$tmpdir/records.td" >&2
  exit 1
fi
if grep -q "Aborted(" "$tmpdir/stdout.log" "$tmpdir/stderr.log"; then
  echo "[wasm-tblgen-hostpath] runtime abort detected" >&2
  cat "$tmpdir/stderr.log" >&2
  exit 1
fi

echo "[wasm-tblgen-hostpath] PASS"

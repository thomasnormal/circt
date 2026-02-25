#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-wasm}"
NODE_BIN="${NODE_BIN:-node}"
BMC_JS="${BMC_JS:-$BUILD_DIR/bin/circt-bmc.js}"
INPUT_MLIR="${INPUT_MLIR:-test/Tools/circt-bmc/disable-iff-const-property-unsat.mlir}"

if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "[wasm-bmc-hostpath] missing Node.js runtime: $NODE_BIN" >&2
  exit 1
fi

if [[ ! -f "$BMC_JS" ]]; then
  echo "[wasm-bmc-hostpath] missing tool: $BMC_JS" >&2
  exit 1
fi

if [[ ! -f "$INPUT_MLIR" ]]; then
  echo "[wasm-bmc-hostpath] missing input: $INPUT_MLIR" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

# Resolve to an absolute host path so the check catches both relative and
# absolute path handling regressions in wasm/node mode.
abs_input="$(cd "$(dirname "$INPUT_MLIR")" && pwd)/$(basename "$INPUT_MLIR")"
out_smt2="$tmpdir/out.smt2"

echo "[wasm-bmc-hostpath] positional host-path input should load in wasm/node mode"
set +e
"$NODE_BIN" "$BMC_JS" \
  --resource-guard=false \
  -b 3 \
  --module m_const_prop \
  --emit-smtlib \
  -o "$out_smt2" \
  "$abs_input" \
  >"$tmpdir/bmc.out" 2>"$tmpdir/bmc.err"
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "[wasm-bmc-hostpath] run failed (rc=$rc)" >&2
  cat "$tmpdir/bmc.err" >&2
  exit 1
fi
if [[ ! -s "$out_smt2" ]]; then
  echo "[wasm-bmc-hostpath] expected output file not found: $out_smt2" >&2
  exit 1
fi
if ! grep -q "(check-sat)" "$out_smt2"; then
  echo "[wasm-bmc-hostpath] output missing SMT check-sat marker" >&2
  cat "$out_smt2" >&2
  exit 1
fi
if grep -q "could not open input file" "$tmpdir/bmc.err"; then
  echo "[wasm-bmc-hostpath] host-path load failed in wasm runtime" >&2
  cat "$tmpdir/bmc.err" >&2
  exit 1
fi
if grep -q "Aborted(" "$tmpdir/bmc.out" "$tmpdir/bmc.err"; then
  echo "[wasm-bmc-hostpath] runtime abort detected" >&2
  cat "$tmpdir/bmc.err" >&2
  exit 1
fi

echo "[wasm-bmc-hostpath] PASS"

#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-wasm}"
NODE_BIN="${NODE_BIN:-node}"
BMC_JS="${BMC_JS:-$BUILD_DIR/bin/circt-bmc.js}"
INPUT_MLIR="${INPUT_MLIR:-test/Tools/circt-bmc/disable-iff-const-property-unsat.mlir}"

if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "[wasm-bmc-stdout-dash] missing Node.js runtime: $NODE_BIN" >&2
  exit 1
fi
if [[ ! -f "$BMC_JS" ]]; then
  echo "[wasm-bmc-stdout-dash] missing tool: $BMC_JS" >&2
  exit 1
fi
if [[ ! -f "$INPUT_MLIR" ]]; then
  echo "[wasm-bmc-stdout-dash] missing input: $INPUT_MLIR" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

echo "[wasm-bmc-stdout-dash] '-o -' should emit SMT-LIB to stdout"
set +e
"$NODE_BIN" "$BMC_JS" \
  --resource-guard=false \
  -b 3 \
  --module m_const_prop \
  --emit-smtlib \
  -o - \
  "$INPUT_MLIR" \
  >"$tmpdir/stdout.log" 2>"$tmpdir/stderr.log"
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "[wasm-bmc-stdout-dash] run failed (rc=$rc)" >&2
  cat "$tmpdir/stderr.log" >&2
  exit 1
fi
if ! grep -q "(check-sat)" "$tmpdir/stdout.log"; then
  echo "[wasm-bmc-stdout-dash] missing SMT-LIB output on stdout for -o -" >&2
  echo "stdout bytes: $(wc -c <"$tmpdir/stdout.log")" >&2
  cat "$tmpdir/stderr.log" >&2
  exit 1
fi
if grep -q "Aborted(" "$tmpdir/stdout.log" "$tmpdir/stderr.log"; then
  echo "[wasm-bmc-stdout-dash] runtime abort detected" >&2
  cat "$tmpdir/stderr.log" >&2
  exit 1
fi

echo "[wasm-bmc-stdout-dash] PASS"

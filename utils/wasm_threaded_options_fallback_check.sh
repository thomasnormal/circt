#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-wasm}"
NODE_BIN="${NODE_BIN:-node}"
SIM_JS="$BUILD_DIR/bin/circt-sim.js"
INPUT_MLIR="${INPUT_MLIR:-test/Tools/circt-sim/llhd-combinational.mlir}"

if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "[wasm-threaded-opts] missing Node.js runtime: $NODE_BIN" >&2
  exit 1
fi

if [[ ! -f "$SIM_JS" ]]; then
  echo "[wasm-threaded-opts] missing tool: $SIM_JS" >&2
  exit 1
fi
if [[ ! -f "$INPUT_MLIR" ]]; then
  echo "[wasm-threaded-opts] missing input: $INPUT_MLIR" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

echo "[wasm-threaded-opts] timeout option should not abort wasm runtime"
set +e
"$NODE_BIN" "$SIM_JS" --resource-guard=false --timeout 5 "$INPUT_MLIR" \
  >"$tmpdir/timeout.out" 2>"$tmpdir/timeout.err"
timeout_rc=$?
set -e
if [[ "$timeout_rc" -ne 0 ]]; then
  echo "[wasm-threaded-opts] --timeout run failed (rc=$timeout_rc)" >&2
  cat "$tmpdir/timeout.err" >&2
  exit 1
fi
if ! grep -q "Simulation completed" "$tmpdir/timeout.out"; then
  echo "[wasm-threaded-opts] --timeout run missing completion output" >&2
  cat "$tmpdir/timeout.out" >&2
  exit 1
fi
if grep -q "Aborted(" "$tmpdir/timeout.out" "$tmpdir/timeout.err"; then
  echo "[wasm-threaded-opts] --timeout triggered runtime abort" >&2
  cat "$tmpdir/timeout.err" >&2
  exit 1
fi

echo "[wasm-threaded-opts] forced parallel option should not abort wasm runtime"
set +e
CIRCT_SIM_EXPERIMENTAL_PARALLEL=1 \
  "$NODE_BIN" "$SIM_JS" --resource-guard=false --parallel 2 "$INPUT_MLIR" \
  >"$tmpdir/parallel.out" 2>"$tmpdir/parallel.err"
parallel_rc=$?
set -e
if [[ "$parallel_rc" -ne 0 ]]; then
  echo "[wasm-threaded-opts] forced --parallel run failed (rc=$parallel_rc)" >&2
  cat "$tmpdir/parallel.err" >&2
  exit 1
fi
if ! grep -q "Simulation completed" "$tmpdir/parallel.out"; then
  echo "[wasm-threaded-opts] forced --parallel run missing completion output" >&2
  cat "$tmpdir/parallel.out" >&2
  exit 1
fi
if grep -q "Aborted(" "$tmpdir/parallel.out" "$tmpdir/parallel.err"; then
  echo "[wasm-threaded-opts] forced --parallel triggered runtime abort" >&2
  cat "$tmpdir/parallel.err" >&2
  exit 1
fi

echo "[wasm-threaded-opts] PASS"

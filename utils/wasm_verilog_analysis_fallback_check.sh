#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-wasm}"
NODE_BIN="${NODE_BIN:-node}"
VERILOG_JS="${VERILOG_JS:-$BUILD_DIR/bin/circt-verilog.js}"

if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "[wasm-verilog-analysis] missing Node.js runtime: $NODE_BIN" >&2
  exit 1
fi

if [[ ! -f "$VERILOG_JS" ]]; then
  echo "[wasm-verilog-analysis] missing tool: $VERILOG_JS" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

cat >"$tmpdir/minimal.sv" <<'SV'
module top;
  logic a;
endmodule
SV

# Full import path (parse + semantic analysis + lowering) should complete in wasm.
echo "[wasm-verilog-analysis] compile path should not abort"
set +e
"$NODE_BIN" "$VERILOG_JS" \
  --resource-guard=false \
  --no-uvm-auto-include \
  --ir-llhd \
  --top top \
  -o "$tmpdir/out.mlir" \
  "$tmpdir/minimal.sv" \
  >"$tmpdir/compile.out" 2>"$tmpdir/compile.err"
compile_rc=$?
set -e
if [[ "$compile_rc" -ne 0 ]]; then
  echo "[wasm-verilog-analysis] compile run failed (rc=$compile_rc)" >&2
  cat "$tmpdir/compile.err" >&2
  exit 1
fi
if [[ ! -s "$tmpdir/out.mlir" ]]; then
  echo "[wasm-verilog-analysis] compile path did not produce IR output" >&2
  exit 1
fi
if ! grep -Eq "(llhd\\.entity|hw\\.module)" "$tmpdir/out.mlir"; then
  echo "[wasm-verilog-analysis] compile output missing expected IR markers" >&2
  cat "$tmpdir/out.mlir" >&2
  exit 1
fi
if grep -q "Aborted(" "$tmpdir/compile.out" "$tmpdir/compile.err"; then
  echo "[wasm-verilog-analysis] compile path triggered wasm abort" >&2
  cat "$tmpdir/compile.err" >&2
  exit 1
fi
if grep -q "thread constructor failed" "$tmpdir/compile.out" "$tmpdir/compile.err"; then
  echo "[wasm-verilog-analysis] compile path triggered thread-constructor failure" >&2
  cat "$tmpdir/compile.err" >&2
  exit 1
fi

# Lint mode should also complete without thread-related aborts.
echo "[wasm-verilog-analysis] lint-only path should not abort"
set +e
"$NODE_BIN" "$VERILOG_JS" \
  --resource-guard=false \
  --no-uvm-auto-include \
  --lint-only \
  "$tmpdir/minimal.sv" \
  >"$tmpdir/lint.out" 2>"$tmpdir/lint.err"
lint_rc=$?
set -e
if [[ "$lint_rc" -ne 0 ]]; then
  echo "[wasm-verilog-analysis] lint-only run failed (rc=$lint_rc)" >&2
  cat "$tmpdir/lint.err" >&2
  exit 1
fi
if grep -q "Aborted(" "$tmpdir/lint.out" "$tmpdir/lint.err"; then
  echo "[wasm-verilog-analysis] lint-only path triggered wasm abort" >&2
  cat "$tmpdir/lint.err" >&2
  exit 1
fi
if grep -q "thread constructor failed" "$tmpdir/lint.out" "$tmpdir/lint.err"; then
  echo "[wasm-verilog-analysis] lint-only path triggered thread-constructor failure" >&2
  cat "$tmpdir/lint.err" >&2
  exit 1
fi

echo "[wasm-verilog-analysis] PASS"

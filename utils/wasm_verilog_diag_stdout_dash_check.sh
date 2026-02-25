#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-wasm}"
NODE_BIN="${NODE_BIN:-node}"
VERILOG_JS="${VERILOG_JS:-$BUILD_DIR/bin/circt-verilog.js}"

if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "[wasm-verilog-diag-stdout] missing Node.js runtime: $NODE_BIN" >&2
  exit 1
fi
if [[ ! -f "$VERILOG_JS" ]]; then
  echo "[wasm-verilog-diag-stdout] missing tool: $VERILOG_JS" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

cat >"$tmpdir/bad.sv" <<'SV'
module top
  logic a;
endmodule
SV

echo "[wasm-verilog-diag-stdout] diagnostics to '-': expect non-zero rc and plain-text output"
set +e
"$NODE_BIN" "$VERILOG_JS" \
  --resource-guard=false \
  --no-color \
  --diagnostic-format=plain \
  --diagnostic-output - \
  --ir-llhd \
  --single-unit \
  --format=sv \
  -o "$tmpdir/out.mlir" \
  "$tmpdir/bad.sv" \
  >"$tmpdir/stdout.log" 2>"$tmpdir/stderr.log"
rc=$?
set -e

if [[ "$rc" -eq 0 ]]; then
  echo "[wasm-verilog-diag-stdout] expected parse failure, got rc=0" >&2
  exit 1
fi
if ! grep -q "error:" "$tmpdir/stdout.log"; then
  echo "[wasm-verilog-diag-stdout] missing diagnostic text on stdout for --diagnostic-output - " >&2
  echo "stdout bytes: $(wc -c <"$tmpdir/stdout.log")" >&2
  cat "$tmpdir/stderr.log" >&2
  exit 1
fi
if grep -q "Aborted(" "$tmpdir/stdout.log" "$tmpdir/stderr.log"; then
  echo "[wasm-verilog-diag-stdout] runtime abort detected" >&2
  cat "$tmpdir/stderr.log" >&2
  exit 1
fi

echo "[wasm-verilog-diag-stdout] PASS"

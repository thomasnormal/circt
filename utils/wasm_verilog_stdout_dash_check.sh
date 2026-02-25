#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-wasm}"
NODE_BIN="${NODE_BIN:-node}"
VERILOG_JS="${VERILOG_JS:-$BUILD_DIR/bin/circt-verilog.js}"

if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "[wasm-verilog-stdout-dash] missing Node.js runtime: $NODE_BIN" >&2
  exit 1
fi
if [[ ! -f "$VERILOG_JS" ]]; then
  echo "[wasm-verilog-stdout-dash] missing tool: $VERILOG_JS" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

cat >"$tmpdir/minimal.sv" <<'SV'
module top;
  logic clk;
  always_ff @(posedge clk) begin
    clk <= ~clk;
  end
endmodule
SV

echo "[wasm-verilog-stdout-dash] host-path input with -o - should emit IR to stdout"
set +e
"$NODE_BIN" "$VERILOG_JS" \
  --resource-guard=false \
  --no-uvm-auto-include \
  --ir-llhd \
  --single-unit \
  --format=sv \
  --top top \
  -o - \
  "$tmpdir/minimal.sv" \
  >"$tmpdir/host.stdout" 2>"$tmpdir/host.stderr"
host_rc=$?
set -e
if [[ "$host_rc" -ne 0 ]]; then
  echo "[wasm-verilog-stdout-dash] host-path run failed (rc=$host_rc)" >&2
  cat "$tmpdir/host.stderr" >&2
  exit 1
fi
if ! grep -Eq "(llhd\\.entity|hw\\.module)" "$tmpdir/host.stdout"; then
  echo "[wasm-verilog-stdout-dash] host-path -o - produced no IR on stdout" >&2
  echo "stdout bytes: $(wc -c <"$tmpdir/host.stdout")" >&2
  cat "$tmpdir/host.stderr" >&2
  exit 1
fi
if grep -q "Aborted(" "$tmpdir/host.stdout" "$tmpdir/host.stderr"; then
  echo "[wasm-verilog-stdout-dash] host-path run aborted" >&2
  cat "$tmpdir/host.stderr" >&2
  exit 1
fi

echo "[wasm-verilog-stdout-dash] stdin input with -o - should emit IR to stdout"
set +e
cat "$tmpdir/minimal.sv" | "$NODE_BIN" "$VERILOG_JS" \
  --resource-guard=false \
  --no-uvm-auto-include \
  --ir-llhd \
  --single-unit \
  --format=sv \
  --top top \
  -o - \
  - \
  >"$tmpdir/stdin.stdout" 2>"$tmpdir/stdin.stderr"
stdin_rc=$?
set -e
if [[ "$stdin_rc" -ne 0 ]]; then
  echo "[wasm-verilog-stdout-dash] stdin run failed (rc=$stdin_rc)" >&2
  cat "$tmpdir/stdin.stderr" >&2
  exit 1
fi
if ! grep -Eq "(llhd\\.entity|hw\\.module)" "$tmpdir/stdin.stdout"; then
  echo "[wasm-verilog-stdout-dash] stdin -o - produced no IR on stdout" >&2
  echo "stdout bytes: $(wc -c <"$tmpdir/stdin.stdout")" >&2
  cat "$tmpdir/stdin.stderr" >&2
  exit 1
fi
if grep -q "Aborted(" "$tmpdir/stdin.stdout" "$tmpdir/stdin.stderr"; then
  echo "[wasm-verilog-stdout-dash] stdin run aborted" >&2
  cat "$tmpdir/stdin.stderr" >&2
  exit 1
fi

echo "[wasm-verilog-stdout-dash] PASS"

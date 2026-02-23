#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-wasm}"
NODE_BIN="${NODE_BIN:-node}"

VERILOG_JS="$BUILD_DIR/bin/circt-verilog.js"
SIM_JS="$BUILD_DIR/bin/circt-sim.js"
SV_TEST_INPUT="${SV_TEST_INPUT:-test/Tools/circt-sim/wasm-uvm-stub-vcd.sv}"
UVM_STUB_PATH="${UVM_STUB_PATH:-integration_test/circt-bmc/Inputs/uvm_stub}"
SV_SIM_TOP="${SV_SIM_TOP:-wasm_uvm_stub_tb}"

if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "[wasm-uvm-vcd] missing Node.js runtime: $NODE_BIN" >&2
  exit 1
fi

for tool in "$VERILOG_JS" "$SIM_JS"; do
  if [[ ! -f "$tool" ]]; then
    echo "[wasm-uvm-vcd] missing wasm tool: $tool" >&2
    exit 1
  fi
done

for input in "$SV_TEST_INPUT" "$UVM_STUB_PATH/uvm_pkg.sv" "$UVM_STUB_PATH/uvm_macros.svh"; do
  if [[ ! -f "$input" ]]; then
    echo "[wasm-uvm-vcd] missing required input: $input" >&2
    exit 1
  fi
done

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

mlir_out="$tmpdir/uvm-stub.mlir"
sim_out="$tmpdir/uvm-stub-sim.out"
sim_err="$tmpdir/uvm-stub-sim.err"
vcd_out="$tmpdir/uvm-stub.vcd"

echo "[wasm-uvm-vcd] frontend: host-path .sv + --uvm-path"
"$NODE_BIN" "$VERILOG_JS" \
  --resource-guard=false \
  --uvm-path "$UVM_STUB_PATH" \
  --ir-llhd \
  --single-unit \
  --format=sv \
  "$SV_TEST_INPUT" \
  -o "$mlir_out"

if [[ ! -s "$mlir_out" ]]; then
  echo "[wasm-uvm-vcd] frontend produced empty IR output: $mlir_out" >&2
  exit 1
fi

if ! grep -Eq "(hw\\.module|llhd\\.entity|llhd\\.process)" "$mlir_out"; then
  echo "[wasm-uvm-vcd] frontend IR output missing expected dialect content" >&2
  exit 1
fi

echo "[wasm-uvm-vcd] sim: UVM-stub bench -> VCD"
"$NODE_BIN" "$SIM_JS" \
  --resource-guard=false \
  --top "$SV_SIM_TOP" \
  --vcd "$vcd_out" \
  "$mlir_out" \
  >"$sim_out" 2>"$sim_err"

grep -q "uvm stub tb start" "$sim_out"
grep -q "Simulation completed" "$sim_out"

if [[ ! -s "$vcd_out" ]]; then
  echo "[wasm-uvm-vcd] expected VCD output not found or empty: $vcd_out" >&2
  exit 1
fi

if ! grep -q '\$enddefinitions' "$vcd_out"; then
  echo "[wasm-uvm-vcd] expected VCD output to include \$enddefinitions: $vcd_out" >&2
  exit 1
fi

if ! grep -q '\$var' "$vcd_out"; then
  echo "[wasm-uvm-vcd] expected VCD output to include \$var declarations: $vcd_out" >&2
  exit 1
fi

echo "[wasm-uvm-vcd] PASS"

#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-wasm}"
NODE_BIN="${NODE_BIN:-node}"
MAX_TIME_FS="${MAX_TIME_FS:-1000000}"

VERILOG_JS="$BUILD_DIR/bin/circt-verilog.js"
SIM_JS="$BUILD_DIR/bin/circt-sim.js"

if [[ ! -s "$VERILOG_JS" ]]; then
  echo "[wasm-uvm-sim] missing wasm frontend artifact: $VERILOG_JS" >&2
  exit 1
fi
if [[ ! -s "$SIM_JS" ]]; then
  echo "[wasm-uvm-sim] missing wasm simulator artifact: $SIM_JS" >&2
  exit 1
fi
if [[ ! -f "lib/Runtime/uvm-core/src/uvm_pkg.sv" ]]; then
  echo "[wasm-uvm-sim] missing UVM source: lib/Runtime/uvm-core/src/uvm_pkg.sv" >&2
  exit 1
fi
if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "[wasm-uvm-sim] missing Node.js runtime: $NODE_BIN" >&2
  exit 1
fi
if [[ ! "$MAX_TIME_FS" =~ ^[0-9]+$ ]]; then
  echo "[wasm-uvm-sim] MAX_TIME_FS must be a numeric integer (got $MAX_TIME_FS)" >&2
  exit 1
fi
if (( MAX_TIME_FS < 1 )); then
  echo "[wasm-uvm-sim] MAX_TIME_FS must be >= 1 (got $MAX_TIME_FS)" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

cat >"$tmpdir/my_test.sv" <<'SV'
import uvm_pkg::*;
`include "uvm_macros.svh"

class my_test extends uvm_test;
  `uvm_component_utils(my_test)
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    `uvm_info("TEST", "Hello from UVM!", UVM_LOW)
    phase.drop_objection(this);
  endtask
endclass
SV

cat >"$tmpdir/tb_top.sv" <<'SV'
import uvm_pkg::*;
`include "uvm_macros.svh"
`include "my_test.sv"

module tb_top;
  initial run_test("my_test");
endmodule
SV

echo "[wasm-uvm-sim] frontend: compile full uvm_pkg sample"
set +e
"$NODE_BIN" "$VERILOG_JS" \
  --resource-guard=false \
  --ir-llhd \
  --timescale 1ns/1ns \
  --single-unit \
  --uvm-path lib/Runtime/uvm-core \
  -I lib/Runtime/uvm-core/src \
  --top tb_top \
  -o "$tmpdir/design.llhd.mlir" \
  "$tmpdir/tb_top.sv" \
  >"$tmpdir/verilog.stdout" 2>"$tmpdir/verilog.stderr"
verilog_rc=$?
set -e
if [[ "$verilog_rc" -ne 0 ]]; then
  echo "[wasm-uvm-sim] circt-verilog failed with rc=$verilog_rc" >&2
  tail -n 120 "$tmpdir/verilog.stderr" >&2 || true
  exit 1
fi
if [[ ! -s "$tmpdir/design.llhd.mlir" ]]; then
  echo "[wasm-uvm-sim] expected lowered MLIR output is missing: $tmpdir/design.llhd.mlir" >&2
  exit 1
fi

echo "[wasm-uvm-sim] simulator: run interpreted mode with max-time=$MAX_TIME_FS fs"
set +e
"$NODE_BIN" "$SIM_JS" \
  --resource-guard=false \
  --mode interpret \
  --top tb_top \
  --max-time="$MAX_TIME_FS" \
  --vcd "$tmpdir/waves.vcd" \
  "$tmpdir/design.llhd.mlir" \
  >"$tmpdir/sim.stdout" 2>"$tmpdir/sim.stderr"
sim_rc=$?
set -e
if [[ "$sim_rc" -ne 0 ]]; then
  echo "[wasm-uvm-sim] circt-sim failed with rc=$sim_rc" >&2
  tail -n 160 "$tmpdir/sim.stderr" >&2 || true
  exit 1
fi

if grep -Eq "RuntimeError:|memory access out of bounds|Aborted\\(" "$tmpdir/sim.stderr"; then
  echo "[wasm-uvm-sim] detected wasm runtime failure signature in simulator stderr" >&2
  tail -n 200 "$tmpdir/sim.stderr" >&2 || true
  exit 1
fi

if ! grep -q "Starting simulation" "$tmpdir/sim.stdout"; then
  echo "[wasm-uvm-sim] expected simulator startup banner missing" >&2
  tail -n 160 "$tmpdir/sim.stdout" >&2 || true
  exit 1
fi
if ! grep -q "UVM_INFO" "$tmpdir/sim.stdout"; then
  echo "[wasm-uvm-sim] expected UVM output missing from simulator stdout" >&2
  tail -n 160 "$tmpdir/sim.stdout" >&2 || true
  exit 1
fi
if ! grep -q "Simulation completed" "$tmpdir/sim.stdout"; then
  echo "[wasm-uvm-sim] expected simulator completion message missing" >&2
  tail -n 160 "$tmpdir/sim.stdout" >&2 || true
  exit 1
fi

if [[ ! -s "$tmpdir/waves.vcd" ]]; then
  echo "[wasm-uvm-sim] expected waveform file missing: $tmpdir/waves.vcd" >&2
  exit 1
fi
if ! grep -q '\$enddefinitions' "$tmpdir/waves.vcd"; then
  echo "[wasm-uvm-sim] waveform is missing \$enddefinitions: $tmpdir/waves.vcd" >&2
  exit 1
fi

echo "[wasm-uvm-sim] PASS"

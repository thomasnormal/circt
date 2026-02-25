#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-wasm}"
NODE_BIN="${NODE_BIN:-node}"

BMC_JS="$BUILD_DIR/bin/circt-bmc.js"
SIM_JS="$BUILD_DIR/bin/circt-sim.js"
VERILOG_JS="$BUILD_DIR/bin/circt-verilog.js"

BMC_TEST_INPUT="${BMC_TEST_INPUT:-test/Tools/circt-bmc/disable-iff-const-property-unsat.mlir}"
SIM_TEST_INPUT="${SIM_TEST_INPUT:-test/Tools/circt-sim/llhd-combinational.mlir}"
SV_TEST_INPUT="${SV_TEST_INPUT:-test/Tools/circt-sim/reject-raw-sv-input.sv}"

if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "[wasm-rg-default] missing Node.js runtime: $NODE_BIN" >&2
  exit 1
fi

if [[ ! -f "$BMC_JS" || ! -f "$SIM_JS" ]]; then
  echo "[wasm-rg-default] missing wasm tools under $BUILD_DIR/bin" >&2
  exit 1
fi

for input in "$BMC_TEST_INPUT" "$SIM_TEST_INPUT"; do
  if [[ ! -f "$input" ]]; then
    echo "[wasm-rg-default] missing test input: $input" >&2
    exit 1
  fi
done

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

echo "[wasm-rg-default] circt-bmc default resource guard"
bmc_out="$tmpdir/bmc.smt2"
set +e
cat "$BMC_TEST_INPUT" | \
  "$NODE_BIN" "$BMC_JS" -b 3 --module m_const_prop --emit-smtlib -o "$bmc_out" - \
  >"$tmpdir/bmc.out" 2>"$tmpdir/bmc.err"
bmc_rc=$?
set -e
if [[ "$bmc_rc" -ne 0 ]]; then
  echo "[wasm-rg-default] circt-bmc failed with default guard (rc=$bmc_rc)" >&2
  cat "$tmpdir/bmc.err" >&2
  exit 1
fi
if ! grep -q "(check-sat)" "$bmc_out"; then
  echo "[wasm-rg-default] circt-bmc missing SMT-LIB output" >&2
  exit 1
fi
if grep -q "Aborted(" "$tmpdir/bmc.out" "$tmpdir/bmc.err"; then
  echo "[wasm-rg-default] circt-bmc hit runtime abort with default guard" >&2
  cat "$tmpdir/bmc.err" >&2
  exit 1
fi

echo "[wasm-rg-default] circt-sim default resource guard"
set +e
cat "$SIM_TEST_INPUT" | \
  "$NODE_BIN" "$SIM_JS" - \
  >"$tmpdir/sim.out" 2>"$tmpdir/sim.err"
sim_rc=$?
set -e
if [[ "$sim_rc" -ne 0 ]]; then
  echo "[wasm-rg-default] circt-sim failed with default guard (rc=$sim_rc)" >&2
  cat "$tmpdir/sim.err" >&2
  exit 1
fi
if ! grep -q "Simulation completed" "$tmpdir/sim.out"; then
  echo "[wasm-rg-default] circt-sim missing completion output" >&2
  exit 1
fi
if grep -q "Aborted(" "$tmpdir/sim.out" "$tmpdir/sim.err"; then
  echo "[wasm-rg-default] circt-sim hit runtime abort with default guard" >&2
  cat "$tmpdir/sim.err" >&2
  exit 1
fi

if [[ -f "$VERILOG_JS" ]]; then
  if [[ ! -f "$SV_TEST_INPUT" ]]; then
    echo "[wasm-rg-default] missing test input: $SV_TEST_INPUT" >&2
    exit 1
  fi
  echo "[wasm-rg-default] circt-verilog default resource guard"
  verilog_mlir="$tmpdir/verilog.mlir"
  set +e
  cat "$SV_TEST_INPUT" | \
    "$NODE_BIN" "$VERILOG_JS" --no-uvm-auto-include --ir-llhd --single-unit --format=sv -o "$verilog_mlir" - \
    >"$tmpdir/verilog.out" 2>"$tmpdir/verilog.err"
  verilog_rc=$?
  set -e
  if [[ "$verilog_rc" -ne 0 ]]; then
    echo "[wasm-rg-default] circt-verilog failed with default guard (rc=$verilog_rc)" >&2
    cat "$tmpdir/verilog.err" >&2
    exit 1
  fi
  if [[ ! -s "$verilog_mlir" ]]; then
    echo "[wasm-rg-default] circt-verilog missing IR output file" >&2
    exit 1
  fi
  if ! grep -q "hw.module" "$verilog_mlir"; then
    echo "[wasm-rg-default] circt-verilog missing IR output" >&2
    exit 1
  fi
  if grep -q "Aborted(" "$tmpdir/verilog.out" "$tmpdir/verilog.err"; then
    echo "[wasm-rg-default] circt-verilog hit runtime abort with default guard" >&2
    cat "$tmpdir/verilog.err" >&2
    exit 1
  fi
fi

echo "[wasm-rg-default] PASS"

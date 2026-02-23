#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-wasm}"
NINJA_JOBS="${NINJA_JOBS:-1}"
NODE_BIN="${NODE_BIN:-node}"
VCD_PATH="${VCD_PATH:-/tmp/circt-wasm-smoke.vcd}"

BMC_JS="$BUILD_DIR/bin/circt-bmc.js"
SIM_JS="$BUILD_DIR/bin/circt-sim.js"

BMC_TEST_INPUT="test/Tools/circt-bmc/disable-iff-const-property-unsat.mlir"
SIM_TEST_INPUT="test/Tools/circt-sim/llhd-combinational.mlir"

if [[ ! -d "$BUILD_DIR" ]]; then
  echo "[wasm-smoke] build directory not found: $BUILD_DIR" >&2
  exit 1
fi

if [[ ! -f "$BMC_TEST_INPUT" || ! -f "$SIM_TEST_INPUT" ]]; then
  echo "[wasm-smoke] required test input file missing" >&2
  exit 1
fi

echo "[wasm-smoke] Building wasm tools (jobs=$NINJA_JOBS)"
ninja -C "$BUILD_DIR" -j "$NINJA_JOBS" circt-bmc circt-sim

if [[ ! -f "$BMC_JS" || ! -f "$SIM_JS" ]]; then
  echo "[wasm-smoke] expected wasm JS outputs are missing" >&2
  echo "  missing: $BMC_JS and/or $SIM_JS" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

echo "[wasm-smoke] Smoke: circt-bmc.js --help"
"$NODE_BIN" "$BMC_JS" --help >"$tmpdir/bmc-help.out" 2>"$tmpdir/bmc-help.err"
if [[ ! -s "$tmpdir/bmc-help.out" ]]; then
  echo "[wasm-smoke] circt-bmc.js --help produced no stdout" >&2
  exit 1
fi

echo "[wasm-smoke] Smoke: circt-sim.js --help"
"$NODE_BIN" "$SIM_JS" --help >"$tmpdir/sim-help.out" 2>"$tmpdir/sim-help.err"
if [[ ! -s "$tmpdir/sim-help.out" ]]; then
  echo "[wasm-smoke] circt-sim.js --help produced no stdout" >&2
  exit 1
fi

echo "[wasm-smoke] Functional: circt-bmc stdin -> SMT-LIB"
cat "$BMC_TEST_INPUT" | \
  "$NODE_BIN" "$BMC_JS" --resource-guard=false -b 3 --module m_const_prop --emit-smtlib -o - - \
  >"$tmpdir/bmc-func.out" 2>"$tmpdir/bmc-func.err"
grep -q "(check-sat)" "$tmpdir/bmc-func.out"

echo "[wasm-smoke] Functional: circt-sim stdin"
cat "$SIM_TEST_INPUT" | \
  "$NODE_BIN" "$SIM_JS" --resource-guard=false - \
  >"$tmpdir/sim-func.out" 2>"$tmpdir/sim-func.err"
grep -q "b=1" "$tmpdir/sim-func.out"
grep -q "b=0" "$tmpdir/sim-func.out"
grep -q "Simulation completed" "$tmpdir/sim-func.out"

echo "[wasm-smoke] Functional: circt-sim --vcd"
rm -f "$VCD_PATH"
cat "$SIM_TEST_INPUT" | \
  "$NODE_BIN" "$SIM_JS" --resource-guard=false --vcd "$VCD_PATH" - \
  >"$tmpdir/sim-vcd.out" 2>"$tmpdir/sim-vcd.err"
grep -q "Wrote waveform" "$tmpdir/sim-vcd.out"
if [[ ! -s "$VCD_PATH" ]]; then
  echo "[wasm-smoke] expected VCD output not found or empty: $VCD_PATH" >&2
  exit 1
fi

echo "[wasm-smoke] Re-entry: circt-sim callMain help -> run"
bmc_reentry_log="$tmpdir/bmc-reentry.log"
sim_reentry_log="$tmpdir/sim-reentry.log"

"$NODE_BIN" utils/wasm_callmain_reentry_check.js "$SIM_JS" \
  --first --help \
  --second --resource-guard=false --vcd /tmp/reentry.vcd "$SIM_TEST_INPUT" \
  --expect-wasm-file-substr /tmp/reentry.vcd "\$enddefinitions" \
  --forbid-substr "Aborted(" \
  >"$sim_reentry_log" 2>&1
if grep -q "InitLLVM was already initialized!" "$sim_reentry_log"; then
  echo "[wasm-smoke] circt-sim same-instance re-entry still hits InitLLVM guard" >&2
  exit 1
fi

echo "[wasm-smoke] Re-entry: circt-bmc callMain help -> run"
"$NODE_BIN" utils/wasm_callmain_reentry_check.js "$BMC_JS" \
  --preload-file "$BMC_TEST_INPUT" /inputs/test.mlir \
  --first --help \
  --second --resource-guard=false -b 3 --module m_const_prop --emit-smtlib -o /out.smt2 /inputs/test.mlir \
  --expect-wasm-file-substr /out.smt2 "(check-sat)" \
  --forbid-substr "Aborted(" \
  >"$bmc_reentry_log" 2>&1
if grep -q "InitLLVM was already initialized!" "$bmc_reentry_log"; then
  echo "[wasm-smoke] circt-bmc same-instance re-entry still hits InitLLVM guard" >&2
  exit 1
fi

echo "[wasm-smoke] Re-entry: circt-sim run -> run"
"$NODE_BIN" utils/wasm_callmain_reentry_check.js "$SIM_JS" \
  --first --resource-guard=false --vcd /tmp/reentry-run1.vcd "$SIM_TEST_INPUT" \
  --second --resource-guard=false --vcd /tmp/reentry-run2.vcd "$SIM_TEST_INPUT" \
  --expect-wasm-file-substr /tmp/reentry-run1.vcd "\$enddefinitions" \
  --expect-wasm-file-substr /tmp/reentry-run2.vcd "\$enddefinitions" \
  --forbid-substr "Aborted(" \
  >"$tmpdir/sim-reentry-run-run.log" 2>&1

echo "[wasm-smoke] Re-entry: circt-bmc run -> run"
"$NODE_BIN" utils/wasm_callmain_reentry_check.js "$BMC_JS" \
  --preload-file "$BMC_TEST_INPUT" /inputs/test.mlir \
  --first --resource-guard=false -b 3 --module m_const_prop --emit-smtlib -o /out1.smt2 /inputs/test.mlir \
  --second --resource-guard=false -b 3 --module m_const_prop --emit-smtlib -o /out2.smt2 /inputs/test.mlir \
  --expect-wasm-file-substr /out1.smt2 "(check-sat)" \
  --expect-wasm-file-substr /out2.smt2 "(check-sat)" \
  --forbid-substr "Aborted(" \
  >"$tmpdir/bmc-reentry-run-run.log" 2>&1

echo "[wasm-smoke] Re-entry: circt-sim plusargs isolation"
BUILD_DIR="$BUILD_DIR" NODE_BIN="$NODE_BIN" utils/wasm_plusargs_reentry_check.sh

if git diff --quiet -- llvm/llvm/cmake/modules/CrossCompile.cmake; then
  echo "[wasm-smoke] CrossCompile.cmake local edits: none"
else
  echo "[wasm-smoke] CrossCompile.cmake local edits: present (remaining work)"
fi

echo "[wasm-smoke] PASS"

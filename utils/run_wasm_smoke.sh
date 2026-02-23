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
"$NODE_BIN" utils/wasm_callmain_reentry_check.js "$SIM_JS" \
  --first --help \
  --second --resource-guard=false "$SIM_TEST_INPUT" \
  --forbid-substr "Aborted("

echo "[wasm-smoke] Re-entry: circt-bmc callMain help -> help"
"$NODE_BIN" utils/wasm_callmain_reentry_check.js "$BMC_JS" \
  --first --help \
  --second --help \
  --forbid-substr "Aborted("

if git diff --quiet -- llvm/llvm/cmake/modules/CrossCompile.cmake; then
  echo "[wasm-smoke] CrossCompile.cmake local edits: none"
else
  echo "[wasm-smoke] CrossCompile.cmake local edits: present (remaining work)"
fi

echo "[wasm-smoke] PASS"

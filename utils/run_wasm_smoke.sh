#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-wasm}"
NINJA_JOBS="${NINJA_JOBS:-1}"
NODE_BIN="${NODE_BIN:-node}"
VCD_PATH="${VCD_PATH:-/tmp/circt-wasm-smoke.vcd}"
WASM_REQUIRE_VERILOG="${WASM_REQUIRE_VERILOG:-0}"
WASM_SKIP_BUILD="${WASM_SKIP_BUILD:-0}"
WASM_CHECK_CXX20_WARNINGS="${WASM_CHECK_CXX20_WARNINGS:-auto}"
WASM_REQUIRE_CLEAN_CROSSCOMPILE="${WASM_REQUIRE_CLEAN_CROSSCOMPILE:-0}"

BMC_JS="$BUILD_DIR/bin/circt-bmc.js"
SIM_JS="$BUILD_DIR/bin/circt-sim.js"
VERILOG_JS="$BUILD_DIR/bin/circt-verilog.js"
BMC_WASM="$BUILD_DIR/bin/circt-bmc.wasm"
SIM_WASM="$BUILD_DIR/bin/circt-sim.wasm"
VERILOG_WASM="$BUILD_DIR/bin/circt-verilog.wasm"

BMC_TEST_INPUT="test/Tools/circt-bmc/disable-iff-const-property-unsat.mlir"
SIM_TEST_INPUT="test/Tools/circt-sim/llhd-combinational.mlir"
SV_TEST_INPUT="test/Tools/circt-sim/reject-raw-sv-input.sv"
SV_SIM_TEST_INPUT="test/Tools/circt-sim/event-triggered.sv"
SV_SIM_TOP="event_triggered_tb"

if [[ ! -d "$BUILD_DIR" ]]; then
  echo "[wasm-smoke] build directory not found: $BUILD_DIR" >&2
  exit 1
fi

if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "[wasm-smoke] missing Node.js runtime: $NODE_BIN" >&2
  exit 1
fi

if ! command -v ninja >/dev/null 2>&1; then
  if [[ "$WASM_SKIP_BUILD" != "1" ]]; then
    echo "[wasm-smoke] missing ninja (required unless WASM_SKIP_BUILD=1)" >&2
    exit 1
  fi
fi

if [[ ! -f "$BMC_TEST_INPUT" || ! -f "$SIM_TEST_INPUT" ]]; then
  echo "[wasm-smoke] required test input file missing" >&2
  exit 1
fi

if [[ ! -f "$SV_TEST_INPUT" || ! -f "$SV_SIM_TEST_INPUT" ]]; then
  echo "[wasm-smoke] required SystemVerilog input file missing" >&2
  exit 1
fi

if [[ "$WASM_CHECK_CXX20_WARNINGS" == "auto" ]]; then
  if [[ "$WASM_SKIP_BUILD" == "1" ]]; then
    WASM_CHECK_CXX20_WARNINGS=0
  else
    WASM_CHECK_CXX20_WARNINGS=1
  fi
fi

if [[ "$WASM_SKIP_BUILD" == "1" ]]; then
  echo "[wasm-smoke] Skipping wasm rebuild (WASM_SKIP_BUILD=1)"
else
  echo "[wasm-smoke] Building wasm tools (jobs=$NINJA_JOBS)"
  ninja -C "$BUILD_DIR" -j "$NINJA_JOBS" circt-bmc circt-sim
fi

if [[ ! -s "$BMC_JS" || ! -s "$SIM_JS" || ! -s "$BMC_WASM" || ! -s "$SIM_WASM" ]]; then
  echo "[wasm-smoke] expected wasm tool outputs are missing or empty" >&2
  echo "  missing: $BMC_JS and/or $SIM_JS and/or $BMC_WASM and/or $SIM_WASM" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

has_verilog_target=0
if ninja -C "$BUILD_DIR" -t targets all >"$tmpdir/targets.list"; then
  if grep -q 'circt-verilog: phony' "$tmpdir/targets.list"; then
    has_verilog_target=1
  fi
fi

if [[ "$has_verilog_target" -eq 1 ]]; then
  if [[ "$WASM_SKIP_BUILD" == "1" ]]; then
    echo "[wasm-smoke] Skipping circt-verilog rebuild (WASM_SKIP_BUILD=1)"
  else
    echo "[wasm-smoke] Building optional wasm frontend: circt-verilog"
    ninja -C "$BUILD_DIR" -j "$NINJA_JOBS" circt-verilog
  fi
  if [[ ! -s "$VERILOG_JS" || ! -s "$VERILOG_WASM" ]]; then
    echo "[wasm-smoke] circt-verilog target exists but expected outputs are missing or empty" >&2
    echo "  missing: $VERILOG_JS and/or $VERILOG_WASM" >&2
    exit 1
  fi
elif [[ "$WASM_REQUIRE_VERILOG" == "1" ]]; then
  echo "[wasm-smoke] circt-verilog target is not configured in $BUILD_DIR" >&2
  echo "  reconfigure with -DCIRCT_SLANG_FRONTEND_ENABLED=ON" >&2
  exit 1
else
  echo "[wasm-smoke] circt-verilog target not configured; skipping SV frontend checks"
fi

if [[ "$has_verilog_target" -eq 0 && ( -f "$VERILOG_JS" || -f "$VERILOG_WASM" ) ]]; then
  echo "[wasm-smoke] found prior circt-verilog artifacts ($VERILOG_JS and/or $VERILOG_WASM); frontend functional checks will still run via default-guard regression"
fi

if [[ "$WASM_CHECK_CXX20_WARNINGS" == "1" ]]; then
  echo "[wasm-smoke] C++20 warning triage"
  if [[ ! -x "utils/wasm_cxx20_warning_check.sh" ]]; then
    echo "[wasm-smoke] missing executable warning check script: utils/wasm_cxx20_warning_check.sh" >&2
    exit 1
  fi
  BUILD_DIR="$BUILD_DIR" NINJA_JOBS="$NINJA_JOBS" utils/wasm_cxx20_warning_check.sh
fi

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

if [[ "$has_verilog_target" -eq 1 ]]; then
  echo "[wasm-smoke] Smoke: circt-verilog.js --help"
  "$NODE_BIN" "$VERILOG_JS" --help >"$tmpdir/verilog-help.out" 2>"$tmpdir/verilog-help.err"
  if [[ ! -s "$tmpdir/verilog-help.out" ]]; then
    echo "[wasm-smoke] circt-verilog.js --help produced no stdout" >&2
    exit 1
  fi

  echo "[wasm-smoke] Functional: circt-verilog stdin (.sv) -> IR"
  cat "$SV_TEST_INPUT" | \
    "$NODE_BIN" "$VERILOG_JS" --resource-guard=false --ir-llhd --single-unit --format=sv - \
    >"$tmpdir/verilog-func.out" 2>"$tmpdir/verilog-func.err"
  grep -Eq "(hw\\.module|llhd\\.entity)" "$tmpdir/verilog-func.out"

  echo "[wasm-smoke] Functional: circt-verilog (.sv) -> circt-sim"
  cat "$SV_SIM_TEST_INPUT" | \
    "$NODE_BIN" "$VERILOG_JS" --resource-guard=false --ir-llhd --single-unit --format=sv - \
    >"$tmpdir/verilog-sim.mlir" 2>"$tmpdir/verilog-sim-verilog.err"
  "$NODE_BIN" "$SIM_JS" --resource-guard=false --top "$SV_SIM_TOP" \
    --vcd "$tmpdir/verilog-sim.vcd" "$tmpdir/verilog-sim.mlir" \
    >"$tmpdir/verilog-sim.out" 2>"$tmpdir/verilog-sim.err"
  grep -q "event triggered ok" "$tmpdir/verilog-sim.out"
  grep -q "Simulation completed" "$tmpdir/verilog-sim.out"
  if [[ ! -s "$tmpdir/verilog-sim.vcd" ]]; then
    echo "[wasm-smoke] expected SV pipeline VCD output not found or empty: $tmpdir/verilog-sim.vcd" >&2
    exit 1
  fi
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

if [[ "$has_verilog_target" -eq 1 ]]; then
  echo "[wasm-smoke] Re-entry: circt-verilog callMain help -> run"
  verilog_reentry_log="$tmpdir/verilog-reentry.log"
  "$NODE_BIN" utils/wasm_callmain_reentry_check.js "$VERILOG_JS" \
    --preload-file "$SV_SIM_TEST_INPUT" /inputs/test.sv \
    --first --help \
    --second --resource-guard=false --ir-llhd --single-unit --format=sv -o /out.mlir /inputs/test.sv \
    --expect-wasm-file-substr /out.mlir "llhd.process" \
    --forbid-substr "Aborted(" \
    >"$verilog_reentry_log" 2>&1
  if grep -q "InitLLVM was already initialized!" "$verilog_reentry_log"; then
    echo "[wasm-smoke] circt-verilog same-instance re-entry still hits InitLLVM guard" >&2
    exit 1
  fi
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

if [[ "$has_verilog_target" -eq 1 ]]; then
  echo "[wasm-smoke] Re-entry: circt-verilog run -> run"
  "$NODE_BIN" utils/wasm_callmain_reentry_check.js "$VERILOG_JS" \
    --preload-file "$SV_SIM_TEST_INPUT" /inputs/test.sv \
    --first --resource-guard=false --ir-hw --single-unit --format=sv -o /out1.mlir /inputs/test.sv \
    --second --resource-guard=false --ir-llhd --single-unit --format=sv -o /out2.mlir /inputs/test.sv \
    --expect-wasm-file-substr /out1.mlir "hw.module" \
    --expect-wasm-file-substr /out2.mlir "llhd.process" \
    --forbid-substr "Aborted(" \
    >"$tmpdir/verilog-reentry-run-run.log" 2>&1
fi

echo "[wasm-smoke] Re-entry: circt-sim plusargs isolation"
BUILD_DIR="$BUILD_DIR" NODE_BIN="$NODE_BIN" utils/wasm_plusargs_reentry_check.sh

echo "[wasm-smoke] Default guard: no wasm runtime abort"
BUILD_DIR="$BUILD_DIR" NODE_BIN="$NODE_BIN" utils/wasm_resource_guard_default_check.sh

if git -C llvm diff --quiet -- llvm/cmake/modules/CrossCompile.cmake; then
  echo "[wasm-smoke] CrossCompile.cmake local edits (llvm submodule): none"
else
  if [[ "$WASM_REQUIRE_CLEAN_CROSSCOMPILE" == "1" ]]; then
    echo "[wasm-smoke] CrossCompile.cmake local edits (llvm submodule): present (failing because WASM_REQUIRE_CLEAN_CROSSCOMPILE=1)" >&2
    exit 1
  fi
  echo "[wasm-smoke] CrossCompile.cmake local edits (llvm submodule): present (remaining work)"
fi

echo "[wasm-smoke] PASS"

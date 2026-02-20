#!/usr/bin/env bash
# Simulate Ibex UVM testbench through circt-verilog + circt-sim.
#
# This script compiles the full Ibex UVM testbench (RTL + DV library + UVM
# agents) through circt-verilog and runs circt-sim with UVM test phases.
#
# The Ibex UVM TB normally requires Spike co-simulation DPI, but circt-sim
# can handle DPI imports via --shared-libs. If the Spike DPI .so is not
# available, the script skips cosim (compile-only or stub mode).
#
# Usage:
#   utils/run_ibex_uvm_circt_sim.sh [out_dir]
#
# Key env vars:
#   IBEX_ROOT=~/ibex                     Path to Ibex checkout
#   CIRCT_SIM_MODE=interpret|compile     (default: interpret)
#   UVM_TESTNAME=core_ibex_base_test     UVM test to run
#   UVM_VERBOSITY=UVM_LOW               Verbosity level
#   SEED=1                              Random seed
#   TIMEOUT=120                         Sim wall-clock timeout (seconds)
#   MAX_CYCLES=100000                   Max simulation cycles
#   COMPILE_ONLY=0                      If 1, only compile (skip sim)
#   SPIKE_DPI_LIB=                      Path to Spike DPI .so (optional)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIRCT_ROOT="${CIRCT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

IBEX_ROOT="${IBEX_ROOT:-$HOME/ibex}"
CIRCT_VERILOG="${CIRCT_VERILOG:-$CIRCT_ROOT/build-test/bin/circt-verilog}"
CIRCT_SIM="${CIRCT_SIM:-$CIRCT_ROOT/build-test/bin/circt-sim}"
OUT_DIR="${1:-/tmp/ibex-uvm-circt-sim-$(date +%Y%m%d-%H%M%S)}"

UVM_DIR="${UVM_DIR:-$CIRCT_ROOT/lib/Runtime/uvm-core/src}"
CIRCT_SIM_MODE="${CIRCT_SIM_MODE:-interpret}"
UVM_TESTNAME="${UVM_TESTNAME:-core_ibex_base_test}"
UVM_VERBOSITY="${UVM_VERBOSITY:-UVM_LOW}"
SEED="${SEED:-1}"
TIMEOUT="${TIMEOUT:-120}"
MAX_CYCLES="${MAX_CYCLES:-100000}"
COMPILE_ONLY="${COMPILE_ONLY:-0}"
SPIKE_DPI_LIB="${SPIKE_DPI_LIB:-}"

mkdir -p "$OUT_DIR"

# Validate tools
for tool in "$CIRCT_VERILOG" "$CIRCT_SIM"; do
  if [[ ! -x "$tool" ]]; then
    echo "error: tool not found or not executable: $tool" >&2
    exit 1
  fi
done

if [[ ! -d "$IBEX_ROOT/rtl" ]]; then
  echo "error: IBEX_ROOT does not look like an Ibex checkout: $IBEX_ROOT" >&2
  exit 1
fi

# Resolve paths
LOWRISC_IP_DIR="$IBEX_ROOT/vendor/lowrisc_ip"
PRJ_DIR="$IBEX_ROOT"

if [[ ! -d "$LOWRISC_IP_DIR" ]]; then
  echo "error: vendor/lowrisc_ip not found in $IBEX_ROOT" >&2
  exit 1
fi

if [[ ! -d "$UVM_DIR" ]]; then
  echo "error: UVM directory not found: $UVM_DIR" >&2
  exit 1
fi

echo "=== Ibex UVM circt-sim Runner ==="
echo "IBEX_ROOT=$IBEX_ROOT"
echo "OUT_DIR=$OUT_DIR"
echo "UVM_TESTNAME=$UVM_TESTNAME"
echo "SEED=$SEED"
echo "CIRCT_SIM_MODE=$CIRCT_SIM_MODE"
echo ""

# ── Step 1: Compile ──────────────────────────────────────────────────────────

MLIR_FILE="$OUT_DIR/ibex_uvm.mlir"
COMPILE_LOG="$OUT_DIR/compile.log"

# Expand the ibex_dv.f filelist with resolved paths.
# We resolve ${PRJ_DIR} and ${LOWRISC_IP_DIR} variable references.
RESOLVED_FILES=()
INCLUDE_DIRS=()

# Parse ibex_dv.f and resolve paths
DV_F="$IBEX_ROOT/dv/uvm/core_ibex/ibex_dv.f"
if [[ ! -f "$DV_F" ]]; then
  echo "error: ibex_dv.f not found: $DV_F" >&2
  exit 1
fi

while IFS= read -r line; do
  # Skip comments and empty lines
  line="${line%%//*}"
  line="${line#"${line%%[![:space:]]*}"}"
  [[ -z "$line" ]] && continue

  # Resolve environment variable references
  line="${line//\$\{PRJ_DIR\}/$PRJ_DIR}"
  line="${line//\$\{LOWRISC_IP_DIR\}/$LOWRISC_IP_DIR}"

  if [[ "$line" == +incdir+* ]]; then
    dir="${line#+incdir+}"
    INCLUDE_DIRS+=("-I" "$dir")
  elif [[ "$line" == +define+* ]]; then
    define="${line#+define+}"
    INCLUDE_DIRS+=("-D$define")
  elif [[ -f "$line" ]]; then
    RESOLVED_FILES+=("$line")
  else
    echo "warning: skipping unresolved line: $line" >&2
  fi
done < "$DV_F"

echo "Parsed ${#RESOLVED_FILES[@]} source files from ibex_dv.f"

# Build compile command
COMPILE_CMD=(
  "$CIRCT_VERILOG"
  "--ir-hw"
  "--timescale=1ns/1ps"
  "-DRVFI"
  "-DUVM"
  "-DTRACE_EXECUTION"
  "${INCLUDE_DIRS[@]}"
  "-I" "$UVM_DIR"
  "-I" "$UVM_DIR/.."
  "${RESOLVED_FILES[@]}"
  "-o" "$MLIR_FILE"
)

echo "Compiling Ibex UVM testbench (${#RESOLVED_FILES[@]} files)..."
compile_start=$(date +%s)
if "${COMPILE_CMD[@]}" > "$COMPILE_LOG" 2>&1; then
  compile_end=$(date +%s)
  echo "Compile OK ($(( compile_end - compile_start ))s): $MLIR_FILE"
  compile_status="OK"
else
  compile_end=$(date +%s)
  echo "COMPILE FAILED ($(( compile_end - compile_start ))s)"
  tail -30 "$COMPILE_LOG"
  compile_status="FAIL"
fi

# Write compile result
echo -e "phase\tstatus\tseconds" > "$OUT_DIR/results.tsv"
echo -e "compile\t${compile_status}\t$(( compile_end - compile_start ))" >> "$OUT_DIR/results.tsv"

if [[ "$compile_status" != "OK" ]]; then
  exit 1
fi

if [[ "$COMPILE_ONLY" == "1" ]]; then
  echo "Compile-only mode; skipping simulation."
  exit 0
fi

# ── Step 2: Simulate ─────────────────────────────────────────────────────────

SIM_LOG="$OUT_DIR/sim.log"
UVM_ARGS="+ntb_random_seed=$SEED +UVM_VERBOSITY=$UVM_VERBOSITY +UVM_TESTNAME=$UVM_TESTNAME"

SIM_CMD=(
  "$CIRCT_SIM"
  "$MLIR_FILE"
  "--top=core_ibex_tb_top"
  "--mode=$CIRCT_SIM_MODE"
  "--timeout=$TIMEOUT"
  "--max-cycles=$MAX_CYCLES"
)

# Add Spike DPI library if available
if [[ -n "$SPIKE_DPI_LIB" && -f "$SPIKE_DPI_LIB" ]]; then
  SIM_CMD+=("--shared-libs=$SPIKE_DPI_LIB")
  echo "Using Spike DPI: $SPIKE_DPI_LIB"
fi

echo "Simulating UVM test: $UVM_TESTNAME (seed=$SEED)..."
sim_start=$(date +%s)
set +e
env CIRCT_UVM_ARGS="$UVM_ARGS" "${SIM_CMD[@]}" > "$SIM_LOG" 2>&1
sim_exit=$?
set -e
sim_end=$(date +%s)
sim_sec=$((sim_end - sim_start))

if [[ $sim_exit -eq 0 ]]; then
  sim_status="PASS"
else
  sim_status="FAIL"
fi

echo "Simulation $sim_status (exit=$sim_exit, ${sim_sec}s)"

# Extract UVM summary if present
if grep -q "UVM_ERROR\|UVM_FATAL" "$SIM_LOG" 2>/dev/null; then
  echo "--- UVM Summary ---"
  grep "UVM_ERROR\|UVM_FATAL\|UVM_WARNING" "$SIM_LOG" | tail -5
fi

echo -e "sim\t${sim_status}\t${sim_sec}" >> "$OUT_DIR/results.tsv"

echo ""
echo "=== Results ==="
cat "$OUT_DIR/results.tsv"

if [[ "$sim_status" != "PASS" ]]; then
  exit 1
fi

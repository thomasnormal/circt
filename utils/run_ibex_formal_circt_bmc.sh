#!/usr/bin/env bash
# Run Ibex formal assertions through circt-bmc.
#
# This script compiles the Ibex RTL (resolved via FuseSoC filelist) through
# circt-verilog and then runs circt-bmc against the embedded SVA assertions.
#
# Usage:
#   utils/run_ibex_formal_circt_bmc.sh [options]
#
# Key env vars:
#   IBEX_ROOT=~/ibex               Path to Ibex checkout
#   CIRCT_VERILOG=build-test/bin/circt-verilog
#   CIRCT_BMC=build-test/bin/circt-bmc
#   BMC_BOUND=20                   BMC unrolling bound
#   TIMEOUT=300                    Wall-clock timeout per target (seconds)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIRCT_ROOT="${CIRCT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

IBEX_ROOT="${IBEX_ROOT:-$HOME/ibex}"
CIRCT_VERILOG="${CIRCT_VERILOG:-$CIRCT_ROOT/build-test/bin/circt-verilog}"
CIRCT_BMC="${CIRCT_BMC:-$CIRCT_ROOT/build-test/bin/circt-bmc}"
OUT_DIR="${1:-/tmp/ibex-formal-circt-bmc-$(date +%Y%m%d-%H%M%S)}"
BMC_BOUND="${BMC_BOUND:-20}"
TIMEOUT="${TIMEOUT:-300}"

mkdir -p "$OUT_DIR"

# Validate tools
for tool in "$CIRCT_VERILOG" "$CIRCT_BMC"; do
  if [[ ! -x "$tool" ]]; then
    echo "error: tool not found or not executable: $tool" >&2
    exit 1
  fi
done

if [[ ! -d "$IBEX_ROOT/rtl" ]]; then
  echo "error: IBEX_ROOT does not look like an Ibex checkout: $IBEX_ROOT" >&2
  exit 1
fi

# Resolve lowRISC IP directory
LOWRISC_IP_DIR="$IBEX_ROOT/vendor/lowrisc_ip"
if [[ ! -d "$LOWRISC_IP_DIR" ]]; then
  echo "error: vendor/lowrisc_ip not found in $IBEX_ROOT" >&2
  exit 1
fi

# Collect RTL source files from ibex_dv.f-style listing.
# We only need the RTL files (not UVM/DV files) for formal.
RTL_FILES=()

# Prim library
RTL_FILES+=(
  "$LOWRISC_IP_DIR/ip/prim_generic/rtl/prim_pkg.sv"
  "$LOWRISC_IP_DIR/ip/prim/rtl/prim_assert.sv"
  "$LOWRISC_IP_DIR/ip/prim/rtl/prim_util_pkg.sv"
  "$LOWRISC_IP_DIR/ip/prim/rtl/prim_count_pkg.sv"
  "$LOWRISC_IP_DIR/ip/prim/rtl/prim_count.sv"
  "$LOWRISC_IP_DIR/ip/prim/rtl/prim_secded_pkg.sv"
  "$LOWRISC_IP_DIR/ip/prim/rtl/prim_mubi_pkg.sv"
  "$LOWRISC_IP_DIR/ip/prim_generic/rtl/prim_ram_1p_pkg.sv"
  "$LOWRISC_IP_DIR/ip/prim/rtl/prim_ram_1p_adv.sv"
  "$LOWRISC_IP_DIR/ip/prim/rtl/prim_ram_1p_scr.sv"
  "$LOWRISC_IP_DIR/ip/prim_generic/rtl/prim_ram_1p.sv"
  "$LOWRISC_IP_DIR/ip/prim_generic/rtl/prim_clock_gating.sv"
  "$LOWRISC_IP_DIR/ip/prim_generic/rtl/prim_buf.sv"
  "$LOWRISC_IP_DIR/ip/prim_generic/rtl/prim_clock_mux2.sv"
  "$LOWRISC_IP_DIR/ip/prim_generic/rtl/prim_flop.sv"
  "$LOWRISC_IP_DIR/ip/prim_generic/rtl/prim_and2.sv"
  "$LOWRISC_IP_DIR/ip/prim/rtl/prim_cipher_pkg.sv"
  "$LOWRISC_IP_DIR/ip/prim/rtl/prim_lfsr.sv"
  "$LOWRISC_IP_DIR/ip/prim/rtl/prim_prince.sv"
  "$LOWRISC_IP_DIR/ip/prim/rtl/prim_subst_perm.sv"
)

# Ibex core RTL
RTL_FILES+=(
  "$IBEX_ROOT/rtl/ibex_pkg.sv"
  "$IBEX_ROOT/rtl/ibex_tracer_pkg.sv"
  "$IBEX_ROOT/rtl/ibex_tracer.sv"
  "$IBEX_ROOT/rtl/ibex_alu.sv"
  "$IBEX_ROOT/rtl/ibex_branch_predict.sv"
  "$IBEX_ROOT/rtl/ibex_compressed_decoder.sv"
  "$IBEX_ROOT/rtl/ibex_controller.sv"
  "$IBEX_ROOT/rtl/ibex_csr.sv"
  "$IBEX_ROOT/rtl/ibex_cs_registers.sv"
  "$IBEX_ROOT/rtl/ibex_counter.sv"
  "$IBEX_ROOT/rtl/ibex_decoder.sv"
  "$IBEX_ROOT/rtl/ibex_dummy_instr.sv"
  "$IBEX_ROOT/rtl/ibex_ex_block.sv"
  "$IBEX_ROOT/rtl/ibex_wb_stage.sv"
  "$IBEX_ROOT/rtl/ibex_id_stage.sv"
  "$IBEX_ROOT/rtl/ibex_icache.sv"
  "$IBEX_ROOT/rtl/ibex_if_stage.sv"
  "$IBEX_ROOT/rtl/ibex_load_store_unit.sv"
  "$IBEX_ROOT/rtl/ibex_lockstep.sv"
  "$IBEX_ROOT/rtl/ibex_multdiv_slow.sv"
  "$IBEX_ROOT/rtl/ibex_multdiv_fast.sv"
  "$IBEX_ROOT/rtl/ibex_prefetch_buffer.sv"
  "$IBEX_ROOT/rtl/ibex_fetch_fifo.sv"
  "$IBEX_ROOT/rtl/ibex_register_file_ff.sv"
  "$IBEX_ROOT/rtl/ibex_pmp.sv"
  "$IBEX_ROOT/rtl/ibex_core.sv"
  "$IBEX_ROOT/rtl/ibex_top.sv"
  "$IBEX_ROOT/rtl/ibex_top_tracing.sv"
)

# Include paths
INCLUDE_DIRS=(
  "-I" "$IBEX_ROOT/rtl"
  "-I" "$LOWRISC_IP_DIR/ip/prim/rtl"
  "-I" "$LOWRISC_IP_DIR/ip/prim_generic/rtl"
  "-I" "$LOWRISC_IP_DIR/dv/sv/dv_utils"
)

# BMC targets: module names with SVA assertions worth checking.
# Start with the core submodules that have the richest assertion sets.
BMC_TARGETS=(
  "ibex_decoder"
  "ibex_controller"
  "ibex_compressed_decoder"
  "ibex_alu"
  "ibex_id_stage"
  "ibex_if_stage"
  "ibex_load_store_unit"
  "ibex_cs_registers"
  "ibex_pmp"
  "ibex_core"
)

echo "=== Ibex Formal BMC Runner ==="
echo "IBEX_ROOT=$IBEX_ROOT"
echo "OUT_DIR=$OUT_DIR"
echo "BMC_BOUND=$BMC_BOUND"
echo "TIMEOUT=$TIMEOUT"
echo "Targets: ${BMC_TARGETS[*]}"
echo ""

# Step 1: Compile all RTL through circt-verilog
MLIR_FILE="$OUT_DIR/ibex_formal.mlir"
COMPILE_LOG="$OUT_DIR/compile.log"

echo "Compiling Ibex RTL..."
COMPILE_CMD=(
  "$CIRCT_VERILOG"
  "--ir-hw"
  "--timescale=1ns/1ps"
  "--no-uvm-auto-include"
  "-DRVFI"
  "${INCLUDE_DIRS[@]}"
  "${RTL_FILES[@]}"
  "-o" "$MLIR_FILE"
)

if ! "${COMPILE_CMD[@]}" > "$COMPILE_LOG" 2>&1; then
  echo "COMPILE FAILED"
  tail -30 "$COMPILE_LOG"
  exit 1
fi
echo "Compile OK: $MLIR_FILE"

# Step 2: Run circt-bmc on each target module
PASS=0
FAIL=0
ERROR=0
RESULTS_FILE="$OUT_DIR/results.tsv"
echo -e "target\tstatus\texit_code\tseconds" > "$RESULTS_FILE"

for target in "${BMC_TARGETS[@]}"; do
  echo -n "BMC $target (bound=$BMC_BOUND)... "
  TARGET_LOG="$OUT_DIR/bmc_${target}.log"
  start_sec=$(date +%s)

  set +e
  timeout "$TIMEOUT" "$CIRCT_BMC" \
    "$MLIR_FILE" \
    --bound="$BMC_BOUND" \
    --top="$target" \
    > "$TARGET_LOG" 2>&1
  rc=$?
  set -e

  end_sec=$(date +%s)
  elapsed=$((end_sec - start_sec))

  if [[ $rc -eq 0 ]]; then
    status="PASS"
    PASS=$((PASS + 1))
  elif [[ $rc -eq 124 ]]; then
    status="TIMEOUT"
    ERROR=$((ERROR + 1))
  else
    status="FAIL"
    FAIL=$((FAIL + 1))
  fi

  echo "$status (${elapsed}s)"
  echo -e "${target}\t${status}\t${rc}\t${elapsed}" >> "$RESULTS_FILE"
done

echo ""
echo "=== Summary ==="
echo "PASS=$PASS  FAIL=$FAIL  ERROR=$ERROR  TOTAL=${#BMC_TARGETS[@]}"
echo "Results: $RESULTS_FILE"

if [[ $FAIL -gt 0 || $ERROR -gt 0 ]]; then
  exit 1
fi

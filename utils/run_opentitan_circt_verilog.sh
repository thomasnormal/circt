#!/usr/bin/env bash
# Parse OpenTitan SystemVerilog files with circt-verilog
set -euo pipefail

usage() {
  echo "usage: $0 <target> [options]"
  echo ""
  echo "Targets (Phase 1 - Primitives):"
  echo "  prim_util_pkg      - Utility package (vbits function)"
  echo "  prim_count_pkg     - Counter package"
  echo "  prim_count         - Hardened counter primitive"
  echo "  prim_flop          - Generic flop primitive"
  echo "  prim_fifo_sync_cnt - FIFO counter logic"
  echo "  prim_fifo_sync     - Synchronous FIFO"
  echo ""
  echo "Targets (Phase 2 - GPIO IP):"
  echo "  tlul_pkg           - TileLink-UL package with dependencies"
  echo "  gpio               - GPIO IP (earlgrey autogen version)"
  echo "  gpio_no_alerts     - GPIO subset without alerts (for HW lowering test)"
  echo ""
  echo "Targets (Phase 3 - TileLink Protocol):"
  echo "  tlul               - TileLink-UL adapter modules"
  echo ""
  echo "Options:"
  echo "  --enable-assertions  Enable assertions (default: disabled)"
  echo "  --ir-hw              Output HW dialect (default: Moore)"
  echo "  --verbose            Verbose output"
  echo "  --dry-run            Print command but don't run"
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

TARGET="$1"
shift

# Parse options
ENABLE_ASSERTIONS=0
IR_OUTPUT="--ir-moore"
VERBOSE=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --enable-assertions) ENABLE_ASSERTIONS=1 ;;
    --ir-hw) IR_OUTPUT="--ir-hw" ;;
    --verbose) VERBOSE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
  shift
done

# Configuration
OPENTITAN_DIR="${OPENTITAN_DIR:-$HOME/opentitan}"
CIRCT_VERILOG="${CIRCT_VERILOG:-build/bin/circt-verilog}"
OUT_DIR="${OUT_DIR:-$PWD}"
TIMESCALE="${TIMESCALE:-1ns/1ps}"

# OpenTitan paths
PRIM_RTL="$OPENTITAN_DIR/hw/ip/prim/rtl"
PRIM_GENERIC_RTL="$OPENTITAN_DIR/hw/ip/prim_generic/rtl"
TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
GPIO_AUTOGEN_RTL="$OPENTITAN_DIR/hw/top_earlgrey/ip_autogen/gpio/rtl"

# Base include paths
INCLUDES=(
  "-I" "$PRIM_RTL"
  "-I" "$PRIM_GENERIC_RTL"
  "-I" "$TLUL_RTL"
  "-I" "$TOP_RTL"
  "-I" "$TOP_AUTOGEN"
  "-I" "$GPIO_AUTOGEN_RTL"
)

# Defines for assertion handling
DEFINES=()
if [[ $ENABLE_ASSERTIONS -eq 0 ]]; then
  # Use dummy assertion macros (same as Verilator)
  DEFINES+=("-DVERILATOR")
fi

# Function to collect files for a target
get_files_for_target() {
  local target=$1
  case "$target" in
    prim_util_pkg)
      echo "$PRIM_RTL/prim_util_pkg.sv"
      ;;
    prim_count_pkg)
      echo "$PRIM_RTL/prim_count_pkg.sv"
      ;;
    prim_flop)
      # Note: prim_assert.sv is `include`d by prim_flop.sv, not a direct source
      echo "$PRIM_RTL/prim_util_pkg.sv"
      echo "$PRIM_GENERIC_RTL/prim_flop.sv"
      ;;
    prim_count)
      # Note: prim_assert.sv is `include`d, not a direct source
      echo "$PRIM_RTL/prim_util_pkg.sv"
      echo "$PRIM_RTL/prim_count_pkg.sv"
      echo "$PRIM_GENERIC_RTL/prim_flop.sv"
      echo "$PRIM_RTL/prim_count.sv"
      ;;
    prim_fifo_sync_cnt)
      # Note: prim_assert.sv is `include`d, not a direct source
      echo "$PRIM_RTL/prim_util_pkg.sv"
      echo "$PRIM_RTL/prim_count_pkg.sv"
      echo "$PRIM_GENERIC_RTL/prim_flop.sv"
      echo "$PRIM_RTL/prim_count.sv"
      echo "$PRIM_RTL/prim_fifo_sync_cnt.sv"
      ;;
    prim_fifo_sync)
      # Note: prim_assert.sv is `include`d, not a direct source
      echo "$PRIM_RTL/prim_util_pkg.sv"
      echo "$PRIM_RTL/prim_count_pkg.sv"
      echo "$PRIM_GENERIC_RTL/prim_flop.sv"
      echo "$PRIM_RTL/prim_count.sv"
      echo "$PRIM_RTL/prim_fifo_sync_cnt.sv"
      echo "$PRIM_RTL/prim_fifo_sync.sv"
      ;;
    tlul_pkg)
      # TileLink-UL package and dependencies
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      echo "$PRIM_RTL/prim_util_pkg.sv"
      echo "$PRIM_RTL/prim_mubi_pkg.sv"
      echo "$PRIM_RTL/prim_secded_pkg.sv"
      echo "$TOP_RTL/top_pkg.sv"
      echo "$OPENTITAN_DIR/hw/ip/tlul/rtl/tlul_pkg.sv"
      ;;
    gpio)
      # Phase 2: GPIO IP (uses earlgrey autogen version)
      # This has many dependencies: TL-UL adapters, subregs, etc.
      local GPIO_RTL="$OPENTITAN_DIR/hw/top_earlgrey/ip_autogen/gpio/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies (in order)
      echo "$PRIM_RTL/prim_util_pkg.sv"
      echo "$PRIM_RTL/prim_mubi_pkg.sv"
      echo "$PRIM_RTL/prim_secded_pkg.sv"
      echo "$TOP_RTL/top_pkg.sv"
      echo "$TLUL_RTL/tlul_pkg.sv"
      echo "$PRIM_RTL/prim_alert_pkg.sv"
      echo "$TOP_AUTOGEN/top_racl_pkg.sv"
      echo "$PRIM_RTL/prim_subreg_pkg.sv"
      # Core primitives
      echo "$PRIM_GENERIC_RTL/prim_flop.sv"
      echo "$PRIM_GENERIC_RTL/prim_buf.sv"
      echo "$PRIM_RTL/prim_cdc_rand_delay.sv"
      echo "$PRIM_GENERIC_RTL/prim_flop_2sync.sv"
      # Security anchor primitives (for prim_alert_sender)
      echo "$PRIM_RTL/prim_sec_anchor_buf.sv"
      echo "$PRIM_RTL/prim_sec_anchor_flop.sv"
      # Filter primitives
      echo "$PRIM_RTL/prim_filter.sv"
      echo "$PRIM_RTL/prim_filter_ctr.sv"
      # Subreg primitives
      echo "$PRIM_RTL/prim_subreg.sv"
      echo "$PRIM_RTL/prim_subreg_ext.sv"
      echo "$PRIM_RTL/prim_subreg_arb.sv"
      echo "$PRIM_RTL/prim_subreg_shadow.sv"
      # Onehot and register check primitives
      echo "$PRIM_RTL/prim_onehot_check.sv"
      echo "$PRIM_RTL/prim_reg_we_check.sv"
      # ECC primitives (secded inverted variants for TL-UL integrity)
      echo "$PRIM_RTL/prim_secded_inv_64_57_dec.sv"
      echo "$PRIM_RTL/prim_secded_inv_64_57_enc.sv"
      echo "$PRIM_RTL/prim_secded_inv_39_32_dec.sv"
      echo "$PRIM_RTL/prim_secded_inv_39_32_enc.sv"
      # Differential decode (for prim_alert_sender)
      echo "$PRIM_RTL/prim_diff_decode.sv"
      # Interrupt and alert primitives
      echo "$PRIM_RTL/prim_intr_hw.sv"
      echo "$PRIM_RTL/prim_alert_sender.sv"
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # GPIO-specific
      echo "$GPIO_RTL/gpio_pkg.sv"
      echo "$GPIO_RTL/gpio_reg_pkg.sv"
      echo "$GPIO_RTL/gpio_reg_top.sv"
      echo "$GPIO_RTL/gpio.sv"
      ;;
    gpio_no_alerts)
      # GPIO subset without alert sender (to test HW lowering without prim_diff_decode)
      # This allows testing if the core GPIO logic can lower to HW dialect
      local GPIO_RTL="$OPENTITAN_DIR/hw/top_earlgrey/ip_autogen/gpio/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies (in order)
      echo "$PRIM_RTL/prim_util_pkg.sv"
      echo "$PRIM_RTL/prim_mubi_pkg.sv"
      echo "$PRIM_RTL/prim_secded_pkg.sv"
      echo "$TOP_RTL/top_pkg.sv"
      echo "$TLUL_RTL/tlul_pkg.sv"
      echo "$PRIM_RTL/prim_alert_pkg.sv"
      echo "$TOP_AUTOGEN/top_racl_pkg.sv"
      echo "$PRIM_RTL/prim_subreg_pkg.sv"
      # Core primitives (without prim_diff_decode dependencies)
      echo "$PRIM_GENERIC_RTL/prim_flop.sv"
      echo "$PRIM_GENERIC_RTL/prim_buf.sv"
      echo "$PRIM_RTL/prim_cdc_rand_delay.sv"
      echo "$PRIM_GENERIC_RTL/prim_flop_2sync.sv"
      # Filter primitives
      echo "$PRIM_RTL/prim_filter.sv"
      echo "$PRIM_RTL/prim_filter_ctr.sv"
      # Subreg primitives
      echo "$PRIM_RTL/prim_subreg.sv"
      echo "$PRIM_RTL/prim_subreg_ext.sv"
      echo "$PRIM_RTL/prim_subreg_arb.sv"
      echo "$PRIM_RTL/prim_subreg_shadow.sv"
      # Onehot and register check primitives
      echo "$PRIM_RTL/prim_onehot_check.sv"
      echo "$PRIM_RTL/prim_reg_we_check.sv"
      # ECC primitives
      echo "$PRIM_RTL/prim_secded_inv_64_57_dec.sv"
      echo "$PRIM_RTL/prim_secded_inv_64_57_enc.sv"
      echo "$PRIM_RTL/prim_secded_inv_39_32_dec.sv"
      echo "$PRIM_RTL/prim_secded_inv_39_32_enc.sv"
      # Interrupt primitive (no alert dependency)
      echo "$PRIM_RTL/prim_intr_hw.sv"
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # GPIO packages (no gpio.sv since it requires prim_alert_sender)
      echo "$GPIO_RTL/gpio_pkg.sv"
      echo "$GPIO_RTL/gpio_reg_pkg.sv"
      echo "$GPIO_RTL/gpio_reg_top.sv"
      # Note: gpio.sv requires prim_alert_sender which requires prim_diff_decode
      ;;
    tlul)
      # Phase 3: TileLink-UL protocol modules
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      # Packages
      echo "$PRIM_RTL/prim_util_pkg.sv"
      echo "$PRIM_RTL/prim_mubi_pkg.sv"
      echo "$PRIM_RTL/prim_secded_pkg.sv"
      echo "$TOP_RTL/top_pkg.sv"
      echo "$TLUL_RTL/tlul_pkg.sv"
      # Primitives
      echo "$PRIM_GENERIC_RTL/prim_flop.sv"
      # TL-UL modules
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      ;;
    *)
      echo "Unknown target: $target" >&2
      return 1
      ;;
  esac
}

# Get files for the target
mapfile -t FILES < <(get_files_for_target "$TARGET")

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No files found for target: $TARGET" >&2
  exit 1
fi

# Check that all files exist
for f in "${FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "File not found: $f" >&2
    exit 1
  fi
done

# Build command
CMD=("$CIRCT_VERILOG")
CMD+=("$IR_OUTPUT")
CMD+=("--timescale=$TIMESCALE")
CMD+=("--no-uvm-auto-include")  # Don't auto-include UVM
CMD+=("${DEFINES[@]}")
CMD+=("${INCLUDES[@]}")
CMD+=("${FILES[@]}")

# Output file
OUT_FILE="$OUT_DIR/opentitan-${TARGET}.log"

if [[ $VERBOSE -eq 1 ]] || [[ $DRY_RUN -eq 1 ]]; then
  echo "Command: ${CMD[*]}"
  echo "Files:"
  for f in "${FILES[@]}"; do
    echo "  - $f"
  done
  echo "Output: $OUT_FILE"
fi

if [[ $DRY_RUN -eq 1 ]]; then
  exit 0
fi

# Run compilation
mkdir -p "$(dirname "$OUT_FILE")"
if "${CMD[@]}" > "$OUT_FILE" 2>&1; then
  echo "SUCCESS: $TARGET parsed"
  echo "Output: $OUT_FILE"
  exit 0
else
  RETVAL=$?
  echo "FAILED: $TARGET (exit code $RETVAL)"
  echo "Output: $OUT_FILE"
  echo ""
  echo "Last 50 lines of output:"
  tail -50 "$OUT_FILE"
  exit $RETVAL
fi

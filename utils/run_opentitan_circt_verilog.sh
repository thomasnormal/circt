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
  echo "Targets (Phase 2 - Peripheral Register Blocks):"
  echo "  gpio_no_alerts     - GPIO register block (without alerts)"
  echo "  uart_reg_top       - UART register block"
  echo "  uart               - Full UART IP with alerts"
  echo "  i2c                - Full I2C IP with alerts"
  echo "  pattgen_reg_top    - Pattgen register block"
  echo "  spi_device         - Full SPI Device IP with alerts"
  echo "  usbdev             - Full USB Device IP with alerts"
  echo "  dma                - Full DMA IP with alerts (multi-port TL-UL)"
  echo "  keymgr_dpe         - Full KeyMgr DPE IP with alerts"
  echo "  rom_ctrl_regs_reg_top - ROM Controller register block"
  echo "  sram_ctrl_regs_reg_top - SRAM Controller register block"
  echo "  sysrst_ctrl_reg_top - System Reset Controller register block (dual clock)"
  echo "  spi_host_reg_top   - SPI Host register block (with TL-UL socket)"
  echo "  spi_host           - Full SPI Host IP with alerts"
  echo "  spi_device_reg_top - SPI Device register block (with 2 TL-UL windows)"
  echo "  i2c_reg_top        - I2C register block"
  echo "  aon_timer_reg_top  - AON Timer register block (dual clock)"
  echo "  pwm_reg_top        - PWM register block (dual clock)"
  echo "  rv_timer_reg_top   - RV Timer register block"
  echo ""
  echo "Targets (Phase 5 - Crypto IPs):"
  echo "  hmac_reg_top       - HMAC crypto register block (with FIFO window)"
  echo "  aes_reg_top        - AES crypto register block (shadowed registers)"
  echo "  csrng_reg_top      - CSRNG crypto register block (random number generator)"
  echo "  edn_reg_top        - EDN crypto register block (entropy distribution network)"
  echo "  keymgr_reg_top     - Key Manager crypto register block (shadowed registers)"
  echo "  kmac_reg_top       - KMAC crypto register block (Keccak MAC with windows)"
  echo "  otbn_reg_top       - OTBN crypto register block (big number accelerator)"
  echo "  otp_ctrl_reg_top   - OTP Controller register block (with window interface)"
  echo "  entropy_src_reg_top - Entropy Source crypto register block (hardware RNG)"
  echo "  lc_ctrl_regs_reg_top - LC Controller register block (lifecycle controller)"
  echo "  flash_ctrl_reg_top - Flash Controller register block (with 2 TL-UL windows)"
  echo "  usbdev_reg_top     - USB Device register block (dual clock, window interface)"
  echo ""
  echo "Targets (Full IP - Working):"
  echo "  rv_timer_full      - Full rv_timer IP with alerts"
  echo "  rv_timer_no_alerts - RV Timer without alerts (timer_core + reg_top)"
  echo "  timer_core         - RISC-V timer core logic only (64-bit mtime/mtimecmp)"
  echo ""
  echo "Targets (TileLink Protocol):"
  echo "  tlul_pkg           - TileLink-UL package with dependencies"
  echo "  tlul               - TileLink-UL adapter modules"
  echo "  gpio               - Full GPIO IP with alerts"
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENTITAN_DIR="${OPENTITAN_DIR:-$HOME/opentitan}"
CIRCT_VERILOG="${CIRCT_VERILOG:-build-test/bin/circt-verilog}"
OUT_DIR="${OUT_DIR:-$PWD}"
TIMESCALE="${TIMESCALE:-1ns/1ps}"

# OpenTitan paths
PRIM_RTL="$OPENTITAN_DIR/hw/ip/prim/rtl"
PRIM_GENERIC_RTL="$OPENTITAN_DIR/hw/ip/prim_generic/rtl"
TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
GPIO_AUTOGEN_RTL="$OPENTITAN_DIR/hw/top_earlgrey/ip_autogen/gpio/rtl"

# Additional OpenTitan paths
UART_RTL="$OPENTITAN_DIR/hw/ip/uart/rtl"
PATTGEN_RTL="$OPENTITAN_DIR/hw/ip/pattgen/rtl"
ROM_CTRL_RTL="$OPENTITAN_DIR/hw/ip/rom_ctrl/rtl"
SRAM_CTRL_RTL="$OPENTITAN_DIR/hw/ip/sram_ctrl/rtl"
SYSRST_CTRL_RTL="$OPENTITAN_DIR/hw/ip/sysrst_ctrl/rtl"
CSRNG_RTL="$OPENTITAN_DIR/hw/ip/csrng/rtl"
EDN_RTL="$OPENTITAN_DIR/hw/ip/edn/rtl"
OTP_CTRL_RTL="$OPENTITAN_DIR/hw/ip/otp_ctrl/rtl"
OTP_CTRL_AUTOGEN_RTL="$OPENTITAN_DIR/hw/top_earlgrey/ip_autogen/otp_ctrl/rtl"
ENTROPY_SRC_RTL="$OPENTITAN_DIR/hw/ip/entropy_src/rtl"
LC_CTRL_RTL="$OPENTITAN_DIR/hw/ip/lc_ctrl/rtl"
USBDEV_RTL="$OPENTITAN_DIR/hw/ip/usbdev/rtl"
DMA_RTL="$OPENTITAN_DIR/hw/ip/dma/rtl"
KEYMGR_RTL="$OPENTITAN_DIR/hw/ip/keymgr/rtl"
KEYMGR_DPE_RTL="$OPENTITAN_DIR/hw/ip/keymgr_dpe/rtl"
KMAC_RTL="$OPENTITAN_DIR/hw/ip/kmac/rtl"

# Base include paths
INCLUDES=(
  "-I" "$PRIM_RTL"
  "-I" "$PRIM_GENERIC_RTL"
  "-I" "$TLUL_RTL"
  "-I" "$TOP_RTL"
  "-I" "$TOP_AUTOGEN"
  "-I" "$GPIO_AUTOGEN_RTL"
  "-I" "$UART_RTL"
  "-I" "$PATTGEN_RTL"
  "-I" "$ROM_CTRL_RTL"
  "-I" "$SRAM_CTRL_RTL"
  "-I" "$SYSRST_CTRL_RTL"
  "-I" "$CSRNG_RTL"
  "-I" "$EDN_RTL"
  "-I" "$OTP_CTRL_RTL"
  "-I" "$OTP_CTRL_AUTOGEN_RTL"
  "-I" "$ENTROPY_SRC_RTL"
  "-I" "$LC_CTRL_RTL"
  "-I" "$USBDEV_RTL"
  "-I" "$DMA_RTL"
  "-I" "$KEYMGR_RTL"
  "-I" "$KEYMGR_DPE_RTL"
  "-I" "$KMAC_RTL"
)

# Defines for assertion handling
DEFINES=()
if [[ $ENABLE_ASSERTIONS -eq 0 ]]; then
  # Use dummy assertion macros (same as Verilator)
  DEFINES+=("-DVERILATOR")
fi

# Target-specific includes
EXTRA_INCLUDES=()
if [[ "$TARGET" == "usbdev" ]]; then
  # Provide a minimal prim_assert shim to avoid macro parsing issues.
  EXTRA_INCLUDES+=("-I" "$SCRIPT_DIR/opentitan_wrappers")
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
    uart_reg_top)
      # UART register block (similar structure to GPIO)
      local UART_RTL="$OPENTITAN_DIR/hw/ip/uart/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies (same as gpio_no_alerts)
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # UART packages
      echo "$UART_RTL/uart_reg_pkg.sv"
      echo "$UART_RTL/uart_reg_top.sv"
      ;;
    uart)
      # Full UART IP (includes UART core, alerts)
      local UART_RTL="$OPENTITAN_DIR/hw/ip/uart/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies (same as uart_reg_top)
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
      # Security anchor primitives
      echo "$PRIM_RTL/prim_sec_anchor_buf.sv"
      echo "$PRIM_RTL/prim_sec_anchor_flop.sv"
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
      # Differential decode and alert sender
      echo "$PRIM_RTL/prim_diff_decode.sv"
      echo "$PRIM_RTL/prim_sec_anchor_buf.sv"
      echo "$PRIM_RTL/prim_sec_anchor_flop.sv"
      echo "$PRIM_RTL/prim_alert_sender.sv"
      # FIFO primitives for UART core
      echo "$PRIM_RTL/prim_count_pkg.sv"
      echo "$PRIM_RTL/prim_count.sv"
      echo "$PRIM_RTL/prim_fifo_sync_cnt.sv"
      echo "$PRIM_RTL/prim_fifo_sync.sv"
      # Interrupt primitive
      echo "$PRIM_RTL/prim_intr_hw.sv"
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # UART packages and IP
      echo "$UART_RTL/uart_reg_pkg.sv"
      echo "$UART_RTL/uart_reg_top.sv"
      echo "$UART_RTL/uart_rx.sv"
      echo "$UART_RTL/uart_tx.sv"
      echo "$UART_RTL/uart_core.sv"
      echo "$UART_RTL/uart.sv"
      ;;
    i2c)
      # Full I2C IP (includes I2C core, alerts)
      local I2C_RTL="$OPENTITAN_DIR/hw/ip/i2c/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies (same as i2c_reg_top)
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
      # Security anchor primitives
      echo "$PRIM_RTL/prim_sec_anchor_buf.sv"
      echo "$PRIM_RTL/prim_sec_anchor_flop.sv"
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
      # Differential decode and alert sender
      echo "$PRIM_RTL/prim_diff_decode.sv"
      echo "$PRIM_RTL/prim_sec_anchor_buf.sv"
      echo "$PRIM_RTL/prim_sec_anchor_flop.sv"
      echo "$PRIM_RTL/prim_alert_sender.sv"
      # FIFO primitives for I2C
      echo "$PRIM_RTL/prim_count_pkg.sv"
      echo "$PRIM_RTL/prim_count.sv"
      echo "$PRIM_RTL/prim_fifo_sync_cnt.sv"
      echo "$PRIM_RTL/prim_fifo_sync.sv"
      # Arbitration and RAM primitives
      echo "$PRIM_RTL/prim_arbiter_tree.sv"
      echo "$PRIM_GENERIC_RTL/prim_ram_1p_pkg.sv"
      echo "$PRIM_GENERIC_RTL/prim_ram_1p.sv"
      echo "$PRIM_RTL/prim_ram_1p_adv.sv"
      # Interrupt primitive
      echo "$PRIM_RTL/prim_intr_hw.sv"
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # I2C packages and IP
      echo "$I2C_RTL/i2c_pkg.sv"
      echo "$I2C_RTL/i2c_reg_pkg.sv"
      echo "$I2C_RTL/i2c_reg_top.sv"
      echo "$I2C_RTL/i2c_fifo_sync_sram_adapter.sv"
      echo "$I2C_RTL/i2c_fifos.sv"
      echo "$I2C_RTL/i2c_bus_monitor.sv"
      echo "$I2C_RTL/i2c_controller_fsm.sv"
      echo "$I2C_RTL/i2c_target_fsm.sv"
      echo "$I2C_RTL/i2c_core.sv"
      echo "$I2C_RTL/i2c.sv"
      ;;
    pattgen_reg_top)
      # Pattgen register block (pattern generator)
      local PATTGEN_RTL="$OPENTITAN_DIR/hw/ip/pattgen/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies (same as uart_reg_top)
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # Pattgen packages
      echo "$PATTGEN_RTL/pattgen_reg_pkg.sv"
      echo "$PATTGEN_RTL/pattgen_reg_top.sv"
      ;;
    rom_ctrl_regs_reg_top)
      # ROM Controller register block (simple register block)
      local ROM_CTRL_RTL="$OPENTITAN_DIR/hw/ip/rom_ctrl/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # ROM Controller packages
      echo "$ROM_CTRL_RTL/rom_ctrl_reg_pkg.sv"
      echo "$ROM_CTRL_RTL/rom_ctrl_regs_reg_top.sv"
      ;;
    sram_ctrl_regs_reg_top)
      # SRAM Controller register block (with RACL support)
      local SRAM_CTRL_RTL="$OPENTITAN_DIR/hw/ip/sram_ctrl/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # SRAM Controller packages
      echo "$SRAM_CTRL_RTL/sram_ctrl_reg_pkg.sv"
      echo "$SRAM_CTRL_RTL/sram_ctrl_regs_reg_top.sv"
      ;;
    sysrst_ctrl_reg_top)
      # System Reset Controller register block (dual clock domain: clk_i and clk_aon_i)
      local SYSRST_CTRL_RTL="$OPENTITAN_DIR/hw/ip/sysrst_ctrl/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # CDC primitives for dual clock domain
      echo "$PRIM_RTL/prim_pulse_sync.sv"
      echo "$PRIM_RTL/prim_sync_reqack.sv"
      echo "$PRIM_RTL/prim_sync_reqack_data.sv"
      echo "$PRIM_RTL/prim_reg_cdc_arb.sv"
      echo "$PRIM_RTL/prim_reg_cdc.sv"
      # System Reset Controller packages
      echo "$SYSRST_CTRL_RTL/sysrst_ctrl_reg_pkg.sv"
      echo "$SYSRST_CTRL_RTL/sysrst_ctrl_reg_top.sv"
      ;;
    spi_host_reg_top)
      # SPI Host register block (similar structure to GPIO/UART)
      local SPI_HOST_RTL="$OPENTITAN_DIR/hw/ip/spi_host/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies (same as gpio_no_alerts/uart_reg_top)
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # TL-UL socket (for multi-window reg tops)
      echo "$PRIM_RTL/prim_count_pkg.sv"
      echo "$PRIM_RTL/prim_count.sv"
      echo "$PRIM_RTL/prim_fifo_sync_cnt.sv"
      echo "$PRIM_RTL/prim_fifo_sync.sv"
      echo "$TLUL_RTL/tlul_fifo_sync.sv"
      echo "$TLUL_RTL/tlul_err_resp.sv"
      echo "$TLUL_RTL/tlul_socket_1n.sv"
      # SPI Host packages
      echo "$SPI_HOST_RTL/spi_host_reg_pkg.sv"
      echo "$SPI_HOST_RTL/spi_host_reg_top.sv"
      ;;
    spi_host)
      # Full SPI Host IP (includes SPI Host core, alerts)
      local SPI_HOST_RTL="$OPENTITAN_DIR/hw/ip/spi_host/rtl"
      local SPI_DEVICE_RTL="$OPENTITAN_DIR/hw/ip/spi_device/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies (same as spi_host_reg_top)
      echo "$PRIM_RTL/prim_assert.sv"
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
      # Security anchor primitives
      echo "$PRIM_RTL/prim_sec_anchor_buf.sv"
      echo "$PRIM_RTL/prim_sec_anchor_flop.sv"
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
      # Differential decode and alert sender
      echo "$PRIM_RTL/prim_diff_decode.sv"
      echo "$PRIM_RTL/prim_sec_anchor_buf.sv"
      echo "$PRIM_RTL/prim_sec_anchor_flop.sv"
      echo "$PRIM_RTL/prim_alert_sender.sv"
      # FIFO primitives for SPI Host
      echo "$PRIM_RTL/prim_count_pkg.sv"
      echo "$PRIM_RTL/prim_count.sv"
      echo "$PRIM_RTL/prim_fifo_sync_cnt.sv"
      echo "$PRIM_RTL/prim_fifo_sync.sv"
      echo "$PRIM_RTL/prim_packer_fifo.sv"
      echo "$PRIM_GENERIC_RTL/prim_flop_en.sv"
      # RACL error arbiter
      echo "$PRIM_RTL/prim_racl_error_arb.sv"
      # Interrupt primitive
      echo "$PRIM_RTL/prim_intr_hw.sv"
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      echo "$TLUL_RTL/tlul_adapter_racl.sv"
      echo "$TLUL_RTL/tlul_adapter_sram.sv"
      echo "$TLUL_RTL/tlul_sram_byte.sv"
      echo "$TLUL_RTL/tlul_adapter_reg_racl.sv"
      echo "$TLUL_RTL/tlul_adapter_sram_racl.sv"
      # TL-UL socket (for multi-window reg tops)
      echo "$TLUL_RTL/tlul_fifo_sync.sv"
      echo "$TLUL_RTL/tlul_err_resp.sv"
      echo "$TLUL_RTL/tlul_socket_1n.sv"
      # SPI packages and IP
      echo "$SPI_DEVICE_RTL/spi_device_reg_pkg.sv"
      echo "$SPI_DEVICE_RTL/spi_device_pkg.sv"
      echo "$SPI_HOST_RTL/spi_host_cmd_pkg.sv"
      echo "$SPI_HOST_RTL/spi_host_reg_pkg.sv"
      echo "$SPI_HOST_RTL/spi_host_reg_top.sv"
      echo "$SPI_HOST_RTL/spi_host_byte_merge.sv"
      echo "$SPI_HOST_RTL/spi_host_byte_select.sv"
      echo "$SPI_HOST_RTL/spi_host_command_queue.sv"
      echo "$SPI_HOST_RTL/spi_host_data_fifos.sv"
      echo "$SCRIPT_DIR/opentitan_wrappers/spi_host_fsm_wrapper.sv"
      echo "$SPI_HOST_RTL/spi_host_shift_register.sv"
      echo "$SPI_HOST_RTL/spi_host_window.sv"
      echo "$SPI_HOST_RTL/spi_host_core.sv"
      echo "$SPI_HOST_RTL/spi_host.sv"
      ;;
    spi_device)
      # Full SPI Device IP (includes SPI Device core, alerts)
      local SPI_DEVICE_RTL="$OPENTITAN_DIR/hw/ip/spi_device/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies (same as spi_device_reg_top)
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
      echo "$PRIM_GENERIC_RTL/prim_flop_en.sv"
      # Clocking and reset primitives
      echo "$PRIM_GENERIC_RTL/prim_clock_gating.sv"
      echo "$PRIM_GENERIC_RTL/prim_clock_inv.sv"
      echo "$PRIM_GENERIC_RTL/prim_clock_mux2.sv"
      echo "$PRIM_GENERIC_RTL/prim_clock_buf.sv"
      echo "$PRIM_GENERIC_RTL/prim_rst_sync.sv"
      # Sync/edge primitives
      echo "$PRIM_RTL/prim_edge_detector.sv"
      echo "$PRIM_RTL/prim_pulse_sync.sv"
      echo "$PRIM_RTL/prim_sync_reqack.sv"
      echo "$PRIM_RTL/prim_sync_reqack_data.sv"
      echo "$PRIM_RTL/prim_slicer.sv"
      # Security anchor primitives
      echo "$PRIM_RTL/prim_sec_anchor_buf.sv"
      echo "$PRIM_RTL/prim_sec_anchor_flop.sv"
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
      # Differential decode and alert sender
      echo "$PRIM_RTL/prim_diff_decode.sv"
      echo "$PRIM_RTL/prim_sec_anchor_buf.sv"
      echo "$PRIM_RTL/prim_sec_anchor_flop.sv"
      echo "$PRIM_RTL/prim_alert_sender.sv"
      # FIFO and SRAM primitives
      echo "$PRIM_RTL/prim_fifo_sync_cnt.sv"
      echo "$PRIM_RTL/prim_fifo_sync.sv"
      echo "$PRIM_RTL/prim_fifo_async.sv"
      echo "$PRIM_RTL/prim_fifo_async_sram_adapter.sv"
      echo "$PRIM_RTL/prim_sram_arbiter.sv"
      echo "$PRIM_RTL/prim_arbiter_ppc.sv"
      echo "$PRIM_RTL/prim_leading_one_ppc.sv"
      # RAM primitives
      echo "$PRIM_GENERIC_RTL/prim_ram_2p_pkg.sv"
      echo "$PRIM_GENERIC_RTL/prim_ram_2p.sv"
      echo "$PRIM_RTL/prim_ram_2p_async_adv.sv"
      echo "$PRIM_RTL/prim_ram_1r1w_async_adv.sv"
      # MUBI sync
      echo "$PRIM_RTL/prim_mubi4_sync.sv"
      # RACL error arbiter
      echo "$PRIM_RTL/prim_racl_error_arb.sv"
      # Interrupt primitive
      echo "$PRIM_RTL/prim_intr_hw.sv"
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      echo "$TLUL_RTL/tlul_adapter_racl.sv"
      echo "$TLUL_RTL/tlul_adapter_sram.sv"
      echo "$TLUL_RTL/tlul_sram_byte.sv"
      echo "$TLUL_RTL/tlul_adapter_sram_racl.sv"
      # TL-UL socket (for multi-window reg tops)
      echo "$TLUL_RTL/tlul_fifo_sync.sv"
      echo "$TLUL_RTL/tlul_err_resp.sv"
      echo "$TLUL_RTL/tlul_socket_1n.sv"
      # SPI device packages and IP
      echo "$SPI_DEVICE_RTL/spi_device_reg_pkg.sv"
      echo "$SPI_DEVICE_RTL/spi_device_pkg.sv"
      echo "$SPI_DEVICE_RTL/spi_device_reg_top.sv"
      echo "$SPI_DEVICE_RTL/spi_s2p.sv"
      echo "$SPI_DEVICE_RTL/spi_p2s.sv"
      echo "$SPI_DEVICE_RTL/spi_cmdparse.sv"
      echo "$SPI_DEVICE_RTL/spi_readcmd.sv"
      echo "$SPI_DEVICE_RTL/spi_passthrough.sv"
      echo "$SPI_DEVICE_RTL/spi_tpm.sv"
      echo "$SPI_DEVICE_RTL/spid_addr_4b.sv"
      echo "$SPI_DEVICE_RTL/spid_csb_sync.sv"
      echo "$SPI_DEVICE_RTL/spid_fifo2sram_adapter.sv"
      echo "$SPI_DEVICE_RTL/spid_dpram.sv"
      echo "$SPI_DEVICE_RTL/spid_jedec.sv"
      echo "$SPI_DEVICE_RTL/spid_readbuffer.sv"
      echo "$SPI_DEVICE_RTL/spid_readsram.sv"
      echo "$SPI_DEVICE_RTL/spid_status.sv"
      echo "$SPI_DEVICE_RTL/spid_upload.sv"
      echo "$SPI_DEVICE_RTL/spi_device.sv"
      ;;
    usbdev)
      # Full USB Device IP (includes USB core, alerts)
      local USBDEV_RTL="$OPENTITAN_DIR/hw/ip/usbdev/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies (same as usbdev_reg_top)
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
      echo "$PRIM_GENERIC_RTL/prim_flop_en.sv"
      # Filter/edge primitives
      echo "$PRIM_RTL/prim_filter.sv"
      echo "$PRIM_RTL/prim_filter_ctr.sv"
      echo "$PRIM_RTL/prim_edge_detector.sv"
      # Clock primitives
      echo "$PRIM_GENERIC_RTL/prim_clock_mux2.sv"
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
      # Differential decode and alert sender
      echo "$PRIM_RTL/prim_diff_decode.sv"
      echo "$PRIM_RTL/prim_sec_anchor_buf.sv"
      echo "$PRIM_RTL/prim_sec_anchor_flop.sv"
      echo "$PRIM_RTL/prim_alert_sender.sv"
      # FIFO primitives
      echo "$PRIM_RTL/prim_fifo_sync_cnt.sv"
      echo "$PRIM_RTL/prim_fifo_sync.sv"
      # CDC primitives
      echo "$PRIM_RTL/prim_reg_cdc.sv"
      echo "$PRIM_RTL/prim_reg_cdc_arb.sv"
      echo "$PRIM_RTL/prim_pulse_sync.sv"
      echo "$PRIM_RTL/prim_sync_reqack.sv"
      # RAM primitives
      echo "$PRIM_GENERIC_RTL/prim_ram_1p_pkg.sv"
      echo "$PRIM_GENERIC_RTL/prim_ram_1p.sv"
      echo "$PRIM_RTL/prim_ram_1p_adv.sv"
      # Interrupt primitive
      echo "$PRIM_RTL/prim_intr_hw.sv"
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      echo "$TLUL_RTL/tlul_adapter_sram.sv"
      echo "$TLUL_RTL/tlul_sram_byte.sv"
      echo "$TLUL_RTL/tlul_err_resp.sv"
      echo "$TLUL_RTL/tlul_fifo_sync.sv"
      echo "$TLUL_RTL/tlul_socket_1n.sv"
      # USB device packages and IP
      echo "$USBDEV_RTL/usb_consts_pkg.sv"
      echo "$USBDEV_RTL/usbdev_pkg.sv"
      echo "$USBDEV_RTL/usbdev_reg_pkg.sv"
      echo "$USBDEV_RTL/usbdev_reg_top.sv"
      echo "$USBDEV_RTL/usbdev_counter.sv"
      echo "$USBDEV_RTL/usbdev_iomux.sv"
      echo "$SCRIPT_DIR/opentitan_wrappers/usbdev_linkstate_wrapper.sv"
      echo "$SCRIPT_DIR/opentitan_wrappers/usbdev_aon_wake_wrapper.sv"
      echo "$SCRIPT_DIR/opentitan_wrappers/usbdev_usbif_wrapper.sv"
      echo "$SCRIPT_DIR/opentitan_wrappers/usb_fs_nb_in_pe_wrapper.sv"
      echo "$SCRIPT_DIR/opentitan_wrappers/usb_fs_nb_out_pe_wrapper.sv"
      echo "$SCRIPT_DIR/opentitan_wrappers/usb_fs_nb_pe_wrapper.sv"
      echo "$USBDEV_RTL/usb_fs_rx.sv"
      echo "$USBDEV_RTL/usb_fs_tx_mux.sv"
      echo "$SCRIPT_DIR/opentitan_wrappers/usb_fs_tx_wrapper.sv"
      echo "$USBDEV_RTL/usbdev.sv"
      ;;
    dma)
      # Full DMA IP
      # Package dependencies
      echo "$PRIM_RTL/prim_util_pkg.sv"
      echo "$PRIM_RTL/prim_mubi_pkg.sv"
      echo "$PRIM_RTL/prim_secded_pkg.sv"
      echo "$TOP_RTL/top_pkg.sv"
      echo "$TLUL_RTL/tlul_pkg.sv"
      echo "$PRIM_RTL/prim_alert_pkg.sv"
      echo "$TOP_AUTOGEN/top_racl_pkg.sv"
      echo "$PRIM_RTL/prim_subreg_pkg.sv"
      echo "$PRIM_RTL/prim_sha2_pkg.sv"
      # Core primitives
      echo "$PRIM_GENERIC_RTL/prim_flop.sv"
      echo "$PRIM_GENERIC_RTL/prim_buf.sv"
      echo "$PRIM_GENERIC_RTL/prim_flop_en.sv"
      echo "$PRIM_GENERIC_RTL/prim_clock_gating.sv"
      echo "$PRIM_RTL/prim_cdc_rand_delay.sv"
      echo "$PRIM_GENERIC_RTL/prim_flop_2sync.sv"
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
      # Differential decode and alert sender
      echo "$PRIM_RTL/prim_diff_decode.sv"
      echo "$PRIM_RTL/prim_sec_anchor_buf.sv"
      echo "$PRIM_RTL/prim_sec_anchor_flop.sv"
      echo "$PRIM_RTL/prim_alert_sender.sv"
      # Sparse FSM primitives
      echo "$PRIM_RTL/prim_sparse_fsm_flop.sv"
      # Interrupt primitive
      echo "$PRIM_RTL/prim_intr_hw.sv"
      # SHA2 primitives
      echo "$PRIM_RTL/prim_sha2_pad.sv"
      echo "$PRIM_RTL/prim_sha2.sv"
      echo "$PRIM_RTL/prim_sha2_32.sv"
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_cmd_intg_gen.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_chk.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      echo "$TLUL_RTL/tlul_adapter_host.sv"
      # DMA packages and IP
      echo "$DMA_RTL/dma_reg_pkg.sv"
      echo "$DMA_RTL/dma_pkg.sv"
      echo "$DMA_RTL/dma_reg_top.sv"
      echo "$DMA_RTL/dma.sv"
      ;;
    keymgr_dpe)
      # Full KeyMgr DPE IP
      # Package dependencies
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
      # Counters and LFSR
      echo "$PRIM_RTL/prim_count_pkg.sv"
      echo "$PRIM_RTL/prim_count.sv"
      echo "$PRIM_RTL/prim_lfsr.sv"
      echo "$PRIM_RTL/prim_sparse_fsm_flop.sv"
      # Mubi + lifecycle sync
      echo "$PRIM_RTL/prim_mubi4_sync.sv"
      echo "$PRIM_RTL/prim_mubi4_sender.sv"
      echo "$PRIM_RTL/prim_lc_sync.sv"
      echo "$PRIM_RTL/prim_edn_req.sv"
      echo "$PRIM_RTL/prim_msb_extend.sv"
      echo "$PRIM_RTL/prim_sync_reqack.sv"
      echo "$PRIM_RTL/prim_sync_reqack_data.sv"
      echo "$PRIM_RTL/prim_packer_fifo.sv"
      echo "$PRIM_RTL/prim_cipher_pkg.sv"
      # Anchor buffers
      echo "$PRIM_RTL/prim_sec_anchor_buf.sv"
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
      # Differential decode and alert sender
      echo "$PRIM_RTL/prim_diff_decode.sv"
      echo "$PRIM_RTL/prim_sec_anchor_buf.sv"
      echo "$PRIM_RTL/prim_sec_anchor_flop.sv"
      echo "$PRIM_RTL/prim_alert_sender.sv"
      # Interrupt primitive
      echo "$PRIM_RTL/prim_intr_hw.sv"
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # Support packages
      echo "$ENTROPY_SRC_RTL/entropy_src_pkg.sv"
      echo "$CSRNG_RTL/csrng_reg_pkg.sv"
      echo "$CSRNG_RTL/csrng_pkg.sv"
      echo "$EDN_RTL/edn_pkg.sv"
      echo "$KMAC_RTL/sha3_pkg.sv"
      echo "$KMAC_RTL/kmac_pkg.sv"
      echo "$OTP_CTRL_RTL/otp_ctrl_pkg.sv"
      echo "$ROM_CTRL_RTL/rom_ctrl_pkg.sv"
      echo "$KEYMGR_RTL/keymgr_reg_pkg.sv"
      echo "$KEYMGR_RTL/keymgr_pkg.sv"
      echo "$LC_CTRL_RTL/lc_ctrl_state_pkg.sv"
      echo "$LC_CTRL_RTL/lc_ctrl_reg_pkg.sv"
      echo "$LC_CTRL_RTL/lc_ctrl_pkg.sv"
      # Keymgr DPE packages and IP
      echo "$KEYMGR_DPE_RTL/keymgr_dpe_pkg.sv"
      echo "$KEYMGR_DPE_RTL/keymgr_dpe_reg_pkg.sv"
      echo "$KEYMGR_DPE_RTL/keymgr_dpe_reg_top.sv"
      echo "$KEYMGR_DPE_RTL/keymgr_dpe_ctrl.sv"
      echo "$KEYMGR_DPE_RTL/keymgr_dpe_op_state_ctrl.sv"
      echo "$KEYMGR_DPE_RTL/keymgr_dpe.sv"
      # Shared keymgr components
      echo "$KEYMGR_RTL/keymgr_cfg_en.sv"
      echo "$KEYMGR_RTL/keymgr_reseed_ctrl.sv"
      echo "$KEYMGR_RTL/keymgr_input_checks.sv"
      echo "$KEYMGR_RTL/keymgr_kmac_if.sv"
      echo "$KEYMGR_RTL/keymgr_sideload_key_ctrl.sv"
      echo "$KEYMGR_RTL/keymgr_sideload_key.sv"
      echo "$KEYMGR_RTL/keymgr_data_en_state.sv"
      echo "$KEYMGR_RTL/keymgr_err.sv"
      ;;
    spi_device_reg_top)
      # SPI Device register block (with 2 window interfaces)
      local SPI_DEVICE_RTL="$OPENTITAN_DIR/hw/ip/spi_device/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies (same as spi_host_reg_top)
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # TL-UL socket (for multi-window reg tops)
      echo "$PRIM_RTL/prim_count_pkg.sv"
      echo "$PRIM_RTL/prim_count.sv"
      echo "$PRIM_RTL/prim_fifo_sync_cnt.sv"
      echo "$PRIM_RTL/prim_fifo_sync.sv"
      echo "$TLUL_RTL/tlul_fifo_sync.sv"
      echo "$TLUL_RTL/tlul_err_resp.sv"
      echo "$TLUL_RTL/tlul_socket_1n.sv"
      # SPI Device packages
      echo "$SPI_DEVICE_RTL/spi_device_reg_pkg.sv"
      echo "$SPI_DEVICE_RTL/spi_device_reg_top.sv"
      ;;
    i2c_reg_top)
      # I2C register block (similar structure to GPIO/UART)
      local I2C_RTL="$OPENTITAN_DIR/hw/ip/i2c/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies (same as gpio_no_alerts/uart_reg_top)
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # I2C packages
      echo "$I2C_RTL/i2c_reg_pkg.sv"
      echo "$I2C_RTL/i2c_reg_top.sv"
      ;;
    aon_timer_reg_top)
      # AON Timer register block
      local AON_TIMER_RTL="$OPENTITAN_DIR/hw/ip/aon_timer/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # CDC primitives for AON timer
      echo "$PRIM_RTL/prim_pulse_sync.sv"
      echo "$PRIM_RTL/prim_sync_reqack.sv"
      echo "$PRIM_RTL/prim_sync_reqack_data.sv"
      echo "$PRIM_RTL/prim_reg_cdc_arb.sv"
      echo "$PRIM_RTL/prim_reg_cdc.sv"
      # AON Timer packages
      echo "$AON_TIMER_RTL/aon_timer_reg_pkg.sv"
      echo "$AON_TIMER_RTL/aon_timer_reg_top.sv"
      ;;
    pwm_reg_top)
      # PWM register block (dual clock domain: clk_i and clk_core_i)
      local PWM_RTL="$OPENTITAN_DIR/hw/top_earlgrey/ip_autogen/pwm/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # CDC primitives for PWM (dual clock domain)
      echo "$PRIM_RTL/prim_pulse_sync.sv"
      echo "$PRIM_RTL/prim_sync_reqack.sv"
      echo "$PRIM_RTL/prim_sync_reqack_data.sv"
      echo "$PRIM_RTL/prim_reg_cdc_arb.sv"
      echo "$PRIM_RTL/prim_reg_cdc.sv"
      # PWM packages
      echo "$PWM_RTL/pwm_reg_pkg.sv"
      echo "$PWM_RTL/pwm_reg_top.sv"
      ;;
    rv_timer_reg_top)
      # RV Timer register block (single clock domain)
      local RV_TIMER_RTL="$OPENTITAN_DIR/hw/ip/rv_timer/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # RV Timer packages
      echo "$RV_TIMER_RTL/rv_timer_reg_pkg.sv"
      echo "$RV_TIMER_RTL/rv_timer_reg_top.sv"
      ;;
    hmac_reg_top)
      # HMAC register block (single clock domain, with FIFO window)
      local HMAC_RTL="$OPENTITAN_DIR/hw/ip/hmac/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters (including socket for window interface)
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # TL-UL socket (for window interface)
      echo "$TLUL_RTL/tlul_err_resp.sv"
      echo "$TLUL_RTL/tlul_fifo_sync.sv"
      # FIFO primitives for socket
      echo "$PRIM_RTL/prim_count_pkg.sv"
      echo "$PRIM_RTL/prim_fifo_sync_cnt.sv"
      echo "$PRIM_RTL/prim_count.sv"
      echo "$PRIM_RTL/prim_fifo_sync.sv"
      echo "$TLUL_RTL/tlul_socket_1n.sv"
      # HMAC packages
      echo "$HMAC_RTL/hmac_reg_pkg.sv"
      echo "$HMAC_RTL/hmac_reg_top.sv"
      ;;
    aes_reg_top)
      # AES register block (crypto IP with shadowed registers)
      local AES_RTL="$OPENTITAN_DIR/hw/ip/aes/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # AES packages
      echo "$AES_RTL/aes_reg_pkg.sv"
      echo "$AES_RTL/aes_reg_top.sv"
      ;;
    csrng_reg_top)
      # CSRNG register block (crypto IP - Cryptographic Secure Random Number Generator)
      local CSRNG_RTL="$OPENTITAN_DIR/hw/ip/csrng/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # CSRNG packages
      echo "$CSRNG_RTL/csrng_reg_pkg.sv"
      echo "$CSRNG_RTL/csrng_reg_top.sv"
      ;;
    edn_reg_top)
      # EDN register block (crypto IP - Entropy Distribution Network)
      local EDN_RTL="$OPENTITAN_DIR/hw/ip/edn/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # EDN packages
      echo "$EDN_RTL/edn_reg_pkg.sv"
      echo "$EDN_RTL/edn_reg_top.sv"
      ;;
    keymgr_reg_top)
      # Key Manager register block (crypto IP with shadowed registers)
      local KEYMGR_RTL="$OPENTITAN_DIR/hw/ip/keymgr/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # Key Manager packages
      echo "$KEYMGR_RTL/keymgr_reg_pkg.sv"
      echo "$KEYMGR_RTL/keymgr_reg_top.sv"
      ;;
    kmac_reg_top)
      # KMAC register block (crypto IP - Keccak Message Authentication Code)
      # Features: shadowed registers, 2 window interfaces for message FIFO and state
      local KMAC_RTL="$OPENTITAN_DIR/hw/ip/kmac/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # TL-UL socket for window interface (KMAC has 2 windows)
      echo "$PRIM_RTL/prim_count_pkg.sv"
      echo "$PRIM_RTL/prim_count.sv"
      echo "$PRIM_RTL/prim_fifo_sync_cnt.sv"
      echo "$PRIM_RTL/prim_fifo_sync.sv"
      echo "$TLUL_RTL/tlul_fifo_sync.sv"
      echo "$TLUL_RTL/tlul_err_resp.sv"
      echo "$TLUL_RTL/tlul_socket_1n.sv"
      # KMAC packages
      echo "$KMAC_RTL/kmac_reg_pkg.sv"
      echo "$KMAC_RTL/kmac_reg_top.sv"
      ;;
    otbn_reg_top)
      # OTBN register block (crypto IP - Big Number Accelerator with window interface)
      local OTBN_RTL="$OPENTITAN_DIR/hw/ip/otbn/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # TL-UL socket for window interface (OTBN has 2 windows)
      echo "$PRIM_RTL/prim_count_pkg.sv"
      echo "$PRIM_RTL/prim_count.sv"
      echo "$PRIM_RTL/prim_fifo_sync_cnt.sv"
      echo "$PRIM_RTL/prim_fifo_sync.sv"
      echo "$TLUL_RTL/tlul_fifo_sync.sv"
      echo "$TLUL_RTL/tlul_err_resp.sv"
      echo "$TLUL_RTL/tlul_socket_1n.sv"
      # OTBN packages
      echo "$OTBN_RTL/otbn_reg_pkg.sv"
      echo "$OTBN_RTL/otbn_reg_top.sv"
      ;;
    entropy_src_reg_top)
      # Entropy Source register block (crypto IP - Hardware Random Number Generator)
      local ENTROPY_SRC_RTL="$OPENTITAN_DIR/hw/ip/entropy_src/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # Entropy Source packages
      echo "$ENTROPY_SRC_RTL/entropy_src_reg_pkg.sv"
      echo "$ENTROPY_SRC_RTL/entropy_src_reg_top.sv"
      ;;
    lc_ctrl_regs_reg_top)
      # LC Controller register block (lifecycle controller)
      local LC_CTRL_RTL="$OPENTITAN_DIR/hw/ip/lc_ctrl/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # LC Controller packages
      echo "$LC_CTRL_RTL/lc_ctrl_reg_pkg.sv"
      echo "$LC_CTRL_RTL/lc_ctrl_regs_reg_top.sv"
      ;;
    flash_ctrl_reg_top)
      # Flash Controller register block (with 2 TL-UL windows for prog/rd FIFOs)
      # Uses flash_ctrl_core_reg_top from autogenerated files
      local FLASH_CTRL_AUTOGEN_RTL="$OPENTITAN_DIR/hw/top_earlgrey/ip_autogen/flash_ctrl/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # TL-UL socket for window interface (flash_ctrl has 2 windows)
      echo "$PRIM_RTL/prim_count_pkg.sv"
      echo "$PRIM_RTL/prim_count.sv"
      echo "$PRIM_RTL/prim_fifo_sync_cnt.sv"
      echo "$PRIM_RTL/prim_fifo_sync.sv"
      echo "$TLUL_RTL/tlul_fifo_sync.sv"
      echo "$TLUL_RTL/tlul_err_resp.sv"
      echo "$TLUL_RTL/tlul_socket_1n.sv"
      # Flash Controller packages
      echo "$FLASH_CTRL_AUTOGEN_RTL/flash_ctrl_reg_pkg.sv"
      echo "$FLASH_CTRL_AUTOGEN_RTL/flash_ctrl_core_reg_top.sv"
      ;;
    otp_ctrl_reg_top)
      # OTP Controller register block (with window interface)
      # Note: Uses otp_ctrl_core_reg_top from autogenerated files
      local OTP_CTRL_RTL="$OPENTITAN_DIR/hw/ip/otp_ctrl/rtl"
      local OTP_CTRL_AUTOGEN_RTL="$OPENTITAN_DIR/hw/top_earlgrey/ip_autogen/otp_ctrl/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies
      echo "$PRIM_RTL/prim_util_pkg.sv"
      echo "$PRIM_RTL/prim_mubi_pkg.sv"
      echo "$PRIM_RTL/prim_secded_pkg.sv"
      echo "$TOP_RTL/top_pkg.sv"
      echo "$TLUL_RTL/tlul_pkg.sv"
      echo "$PRIM_RTL/prim_alert_pkg.sv"
      echo "$TOP_AUTOGEN/top_racl_pkg.sv"
      echo "$PRIM_RTL/prim_subreg_pkg.sv"
      # LC Controller packages (needed by otp_ctrl_pkg)
      local LC_CTRL_RTL="$OPENTITAN_DIR/hw/ip/lc_ctrl/rtl"
      echo "$LC_CTRL_RTL/lc_ctrl_reg_pkg.sv"
      echo "$LC_CTRL_RTL/lc_ctrl_state_pkg.sv"
      echo "$LC_CTRL_RTL/lc_ctrl_pkg.sv"
      # OTP controller packages (needed before reg_top)
      echo "$OTP_CTRL_RTL/otp_ctrl_pkg.sv"
      echo "$OTP_CTRL_AUTOGEN_RTL/otp_ctrl_macro_pkg.sv"
      echo "$OTP_CTRL_AUTOGEN_RTL/otp_ctrl_reg_pkg.sv"
      echo "$OTP_CTRL_AUTOGEN_RTL/otp_ctrl_part_pkg.sv"
      echo "$OTP_CTRL_AUTOGEN_RTL/otp_ctrl_top_specific_pkg.sv"
      # Core primitives
      echo "$PRIM_GENERIC_RTL/prim_flop.sv"
      echo "$PRIM_GENERIC_RTL/prim_buf.sv"
      echo "$PRIM_RTL/prim_cdc_rand_delay.sv"
      echo "$PRIM_GENERIC_RTL/prim_flop_2sync.sv"
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # TL-UL socket for window interface
      echo "$PRIM_RTL/prim_count_pkg.sv"
      echo "$PRIM_RTL/prim_count.sv"
      echo "$PRIM_RTL/prim_fifo_sync_cnt.sv"
      echo "$PRIM_RTL/prim_fifo_sync.sv"
      echo "$TLUL_RTL/tlul_fifo_sync.sv"
      echo "$TLUL_RTL/tlul_err_resp.sv"
      echo "$TLUL_RTL/tlul_socket_1n.sv"
      # OTP Controller core register top
      echo "$OTP_CTRL_AUTOGEN_RTL/otp_ctrl_core_reg_top.sv"
      ;;
    usbdev_reg_top)
      # USB Device register block (dual clock domain: clk_i and clk_aon_i)
      # Features: window interface for buffer memory access, CDC crossing
      local USBDEV_RTL="$OPENTITAN_DIR/hw/ip/usbdev/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # TL-UL socket for window interface
      echo "$PRIM_RTL/prim_count_pkg.sv"
      echo "$PRIM_RTL/prim_count.sv"
      echo "$PRIM_RTL/prim_fifo_sync_cnt.sv"
      echo "$PRIM_RTL/prim_fifo_sync.sv"
      echo "$TLUL_RTL/tlul_fifo_sync.sv"
      echo "$TLUL_RTL/tlul_err_resp.sv"
      echo "$TLUL_RTL/tlul_socket_1n.sv"
      # CDC primitives for dual clock domain
      echo "$PRIM_RTL/prim_pulse_sync.sv"
      echo "$PRIM_RTL/prim_sync_reqack.sv"
      echo "$PRIM_RTL/prim_sync_reqack_data.sv"
      echo "$PRIM_RTL/prim_reg_cdc_arb.sv"
      echo "$PRIM_RTL/prim_reg_cdc.sv"
      # USB Device packages
      echo "$USBDEV_RTL/usbdev_reg_pkg.sv"
      echo "$USBDEV_RTL/usbdev_reg_top.sv"
      ;;
    rv_timer_full)
      # RV Timer full IP (register block + timer_core + alerts)
      # Note: Requires prim_alert_sender which depends on prim_diff_decode
      local RV_TIMER_RTL="$OPENTITAN_DIR/hw/ip/rv_timer/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # Interrupt primitive
      echo "$PRIM_RTL/prim_intr_hw.sv"
      # Alert sender and dependencies (blocked by prim_diff_decode)
      echo "$PRIM_RTL/prim_sec_anchor_buf.sv"
      echo "$PRIM_RTL/prim_sec_anchor_flop.sv"
      echo "$PRIM_RTL/prim_filter.sv"
      echo "$PRIM_RTL/prim_filter_ctr.sv"
      echo "$PRIM_RTL/prim_diff_decode.sv"
      echo "$PRIM_RTL/prim_alert_sender.sv"
      # RV Timer packages and modules
      echo "$RV_TIMER_RTL/rv_timer_reg_pkg.sv"
      echo "$RV_TIMER_RTL/rv_timer_reg_top.sv"
      echo "$RV_TIMER_RTL/timer_core.sv"
      echo "$RV_TIMER_RTL/rv_timer.sv"
      ;;
    rv_timer_no_alerts)
      # RV Timer without prim_alert_sender (workaround for prim_diff_decode bug)
      # Includes timer_core but needs modified rv_timer.sv or stub
      local RV_TIMER_RTL="$OPENTITAN_DIR/hw/ip/rv_timer/rtl"
      local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
      local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
      local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
      # Package dependencies
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
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # Interrupt primitive (for timer interrupts)
      echo "$PRIM_RTL/prim_intr_hw.sv"
      # RV Timer packages and logic (no alert_sender)
      echo "$RV_TIMER_RTL/rv_timer_reg_pkg.sv"
      echo "$RV_TIMER_RTL/rv_timer_reg_top.sv"
      echo "$RV_TIMER_RTL/timer_core.sv"
      # Note: Cannot include rv_timer.sv directly as it requires prim_alert_sender
      # Would need a modified version without alerts for full IP testing
      ;;
    timer_core)
      # timer_core - RISC-V timer core logic (minimal dependencies, 64-bit mtime/mtimecmp)
      # Used to test 64-bit APInt handling in circt-sim
      local RV_TIMER_RTL="$OPENTITAN_DIR/hw/ip/rv_timer/rtl"
      echo "$RV_TIMER_RTL/timer_core.sv"
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
CMD+=("${EXTRA_INCLUDES[@]}")
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

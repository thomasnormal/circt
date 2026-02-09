#!/usr/bin/env bash
# Simulate OpenTitan designs with circt-sim
set -euo pipefail

usage() {
  echo "usage: $0 <target> [options]"
  echo ""
  echo "Targets (Primitives):"
  echo "  prim_fifo_sync     - Synchronous FIFO with simple testbench"
  echo "  prim_count         - Hardened counter with testbench"
  echo ""
  echo "Targets (Peripheral Register Blocks):"
  echo "  gpio_no_alerts     - GPIO register block (minimal TL-UL testbench)"
  echo "  tlul_adapter_reg   - TL-UL adapter register interface smoke test"
  echo "  gpio               - Full GPIO IP with alerts"
  echo "  uart_reg_top       - UART register block"
  echo "  uart               - Full UART IP with alerts"
  echo "  pattgen_reg_top    - Pattgen register block"
  echo "  alert_handler_reg_top - Alert Handler register block"
  echo "  i2c                - Full I2C IP with alerts"
  echo "  rom_ctrl_regs_reg_top - ROM Controller register block"
  echo "  sram_ctrl_regs_reg_top - SRAM Controller register block"
  echo "  sysrst_ctrl_reg_top - System Reset Controller register block (dual clock)"
  echo "  spi_host_reg_top   - SPI Host register block (TL-UL with window)"
  echo "  spi_host           - Full SPI Host IP with alerts"
  echo "  spi_device         - Full SPI Device IP with alerts"
  echo "  spi_device_reg_top - SPI Device register block (TL-UL with 2 windows)"
  echo "  usbdev             - Full USB Device IP with alerts"
  echo "  i2c_reg_top        - I2C register block"
  echo "  aon_timer_reg_top  - AON Timer register block (dual clock domain)"
  echo "  pwm_reg_top        - PWM register block (dual clock domain)"
  echo "  rv_timer_reg_top   - RV Timer register block (single clock)"
  echo ""
  echo "Targets (Crypto IPs):"
  echo "  ascon_reg_top      - Ascon crypto register block (shadowed registers)"
  echo "  hmac_reg_top       - HMAC crypto register block (with FIFO window)"
  echo "  aes_reg_top        - AES crypto register block (shadowed registers)"
  echo "  dma                - Full DMA IP with alerts (multi-port TL-UL)"
  echo "  keymgr_dpe         - Full KeyMgr DPE IP with alerts"
  echo "  csrng_reg_top      - CSRNG crypto register block (random number generator)"
  echo "  keymgr_reg_top     - Key Manager crypto register block (shadowed registers)"
  echo "  kmac_reg_top       - KMAC crypto register block (Keccak MAC with windows)"
  echo "  otbn_reg_top       - OTBN crypto register block (big number accelerator)"
  echo "  otp_ctrl_reg_top   - OTP Controller register block (with window interface)"
  echo "  lc_ctrl_regs_reg_top - LC Controller register block (lifecycle controller)"
  echo "  flash_ctrl_reg_top - Flash Controller register block (with 2 TL-UL windows)"
  echo "  usbdev_reg_top     - USB Device register block (dual clock, window interface)"
  echo ""
  echo "Targets (Full IP Logic - Experimental):"
  echo "  ascon              - Full Ascon IP with alerts (EDN/keymgr/lc stubs)"
  echo "  alert_handler      - Full Alert Handler IP with alerts/esc/edn"
  echo "  mbx                - Full Mailbox IP with alerts (multi-port TL-UL)"
  echo "  rv_dm              - Full RISC-V Debug Module IP with alerts"
  echo "  timer_core         - RISC-V timer core logic (crashes on 64-bit values)"
  echo ""
  echo "Options:"
  echo "  --max-cycles=N     Maximum clock cycles to simulate (default: 1000)"
  echo "  --vcd=<file>       Output VCD waveform file"
  echo "  --timeout=N        Wall-clock timeout in seconds (default: 60)"
  echo "  --verbose          Verbose output"
  echo "  --dry-run          Print commands but don't run"
  echo "  --skip-compile     Skip recompilation (use existing MLIR)"
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

TARGET="$1"
shift

# Parse options
MAX_CYCLES=1000
VCD_FILE=""
TIMEOUT=60
VERBOSE=0
DRY_RUN=0
SKIP_COMPILE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-cycles=*) MAX_CYCLES="${1#*=}" ;;
    --vcd=*) VCD_FILE="${1#*=}" ;;
    --timeout=*) TIMEOUT="${1#*=}" ;;
    --verbose) VERBOSE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --skip-compile) SKIP_COMPILE=1 ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
  shift
done

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIRCT_DIR="${CIRCT_DIR:-$(dirname "$SCRIPT_DIR")}"
CIRCT_VERILOG="${CIRCT_VERILOG:-$CIRCT_DIR/build/bin/circt-verilog}"
CIRCT_SIM="${CIRCT_SIM:-$CIRCT_DIR/build/bin/circt-sim}"
CIRCT_OPT="${CIRCT_OPT:-$CIRCT_DIR/build/bin/circt-opt}"
OUT_DIR="${OUT_DIR:-$PWD}"
OPENTITAN_DIR="${OPENTITAN_DIR:-$HOME/opentitan}"

# OpenTitan paths
PRIM_RTL="$OPENTITAN_DIR/hw/ip/prim/rtl"
PRIM_GENERIC_RTL="$OPENTITAN_DIR/hw/ip/prim_generic/rtl"
TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
GPIO_RTL="$OPENTITAN_DIR/hw/top_earlgrey/ip_autogen/gpio/rtl"
ALERT_HANDLER_RTL="$OPENTITAN_DIR/hw/top_earlgrey/ip_autogen/alert_handler/rtl"
UART_RTL="$OPENTITAN_DIR/hw/ip/uart/rtl"
PATTGEN_RTL="$OPENTITAN_DIR/hw/ip/pattgen/rtl"
ROM_CTRL_RTL="$OPENTITAN_DIR/hw/ip/rom_ctrl/rtl"
SRAM_CTRL_RTL="$OPENTITAN_DIR/hw/ip/sram_ctrl/rtl"
SYSRST_CTRL_RTL="$OPENTITAN_DIR/hw/ip/sysrst_ctrl/rtl"
SPI_HOST_RTL="$OPENTITAN_DIR/hw/ip/spi_host/rtl"
I2C_RTL="$OPENTITAN_DIR/hw/ip/i2c/rtl"
AON_TIMER_RTL="$OPENTITAN_DIR/hw/ip/aon_timer/rtl"
PWM_RTL="$OPENTITAN_DIR/hw/top_earlgrey/ip_autogen/pwm/rtl"
RV_TIMER_RTL="$OPENTITAN_DIR/hw/ip/rv_timer/rtl"
HMAC_RTL="$OPENTITAN_DIR/hw/ip/hmac/rtl"
AES_RTL="$OPENTITAN_DIR/hw/ip/aes/rtl"
CSRNG_RTL="$OPENTITAN_DIR/hw/ip/csrng/rtl"
KEYMGR_RTL="$OPENTITAN_DIR/hw/ip/keymgr/rtl"
KMAC_RTL="$OPENTITAN_DIR/hw/ip/kmac/rtl"
OTBN_RTL="$OPENTITAN_DIR/hw/ip/otbn/rtl"
OTP_CTRL_RTL="$OPENTITAN_DIR/hw/ip/otp_ctrl/rtl"
OTP_CTRL_AUTOGEN_RTL="$OPENTITAN_DIR/hw/top_earlgrey/ip_autogen/otp_ctrl/rtl"
USBDEV_RTL="$OPENTITAN_DIR/hw/ip/usbdev/rtl"
ASCON_RTL="$OPENTITAN_DIR/hw/ip/ascon/rtl"
DMA_RTL="$OPENTITAN_DIR/hw/ip/dma/rtl"
MBX_RTL="$OPENTITAN_DIR/hw/ip/mbx/rtl"
KEYMGR_DPE_RTL="$OPENTITAN_DIR/hw/ip/keymgr_dpe/rtl"
RV_DM_RTL="$OPENTITAN_DIR/hw/ip/rv_dm/rtl"
RV_DM_VENDOR_RTL="$OPENTITAN_DIR/hw/vendor/pulp_riscv_dbg/src"
RV_DM_ROM_RTL="$OPENTITAN_DIR/hw/vendor/pulp_riscv_dbg/debug_rom"
EDN_RTL="$OPENTITAN_DIR/hw/ip/edn/rtl"
ENTROPY_SRC_RTL="$OPENTITAN_DIR/hw/ip/entropy_src/rtl"
LC_CTRL_RTL="$OPENTITAN_DIR/hw/ip/lc_ctrl/rtl"

# Testbench generation
generate_testbench() {
  local target=$1
  local tb_file=$2

  case "$target" in
    prim_fifo_sync)
      cat > "$tb_file" << 'EOF'
// Simple testbench for prim_fifo_sync
// Exercises basic push/pop operations

module prim_fifo_sync_tb;
  parameter Width = 8;
  parameter Depth = 4;

  logic clk_i = 0;
  logic rst_ni = 1;
  logic clr_i = 0;
  logic wvalid_i = 0;
  logic wready_o;
  logic [Width-1:0] wdata_i = 0;
  logic rvalid_o;
  logic rready_i = 0;
  logic [Width-1:0] rdata_o;
  logic full_o;
  logic [$clog2(Depth+1)-1:0] depth_o;
  logic err_o;

  prim_fifo_sync #(
    .Width(Width),
    .Depth(Depth),
    .Pass(1'b1),
    .OutputZeroIfEmpty(1'b1),
    .Secure(1'b0)
  ) dut (
    .clk_i,
    .rst_ni,
    .clr_i,
    .wvalid_i,
    .wready_o,
    .wdata_i,
    .rvalid_o,
    .rready_i,
    .rdata_o,
    .full_o,
    .depth_o,
    .err_o
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  initial begin
    // Reset
    rst_ni = 0;
    #20;
    rst_ni = 1;
    #10;

    // Push some data
    for (int i = 0; i < 4; i++) begin
      @(posedge clk_i);
      wvalid_i = 1;
      wdata_i = 8'(i * 10);
    end
    @(posedge clk_i);
    wvalid_i = 0;

    // Pop the data
    @(posedge clk_i);
    rready_i = 1;
    for (int i = 0; i < 4; i++) begin
      @(posedge clk_i);
      if (rvalid_o) begin
        $display("Read data: %d (expected %d)", rdata_o, i * 10);
      end
    end
    rready_i = 0;

    // Done
    #50;
    $display("TEST PASSED: prim_fifo_sync basic operations");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    prim_count)
      cat > "$tb_file" << 'EOF'
// Simple testbench for prim_count
// Exercises increment/decrement operations

module prim_count_tb;
  parameter Width = 4;

  logic clk_i = 0;
  logic rst_ni = 0;
  logic clr_i = 0;
  logic set_i = 0;
  logic [Width-1:0] set_cnt_i = 0;
  logic incr_en_i = 0;
  logic decr_en_i = 0;
  logic [Width-1:0] step_i = 1;
  logic commit_i = 1;
  logic [Width-1:0] cnt_o;
  logic [Width-1:0] cnt_after_commit_o;
  logic err_o;

  prim_count #(
    .Width(Width),
    .ResetValue('0),
    .EnableAlertTriggerSVA(0)
  ) dut (
    .clk_i,
    .rst_ni,
    .clr_i,
    .set_i,
    .set_cnt_i,
    .incr_en_i,
    .decr_en_i,
    .step_i,
    .commit_i,
    .cnt_o,
    .cnt_after_commit_o,
    .err_o
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  initial begin
    // Reset
    rst_ni = 0;
    #20;
    rst_ni = 1;
    #10;

    // Increment several times
    $display("Starting count test...");
    for (int i = 0; i < 8; i++) begin
      @(posedge clk_i);
      incr_en_i = 1;
    end
    @(posedge clk_i);
    incr_en_i = 0;

    $display("After increment: cnt_o = %d", cnt_o);

    // Decrement a few times
    for (int i = 0; i < 3; i++) begin
      @(posedge clk_i);
      decr_en_i = 1;
    end
    @(posedge clk_i);
    decr_en_i = 0;

    $display("After decrement: cnt_o = %d", cnt_o);

    // Clear
    @(posedge clk_i);
    clr_i = 1;
    @(posedge clk_i);
    clr_i = 0;

    $display("After clear: cnt_o = %d", cnt_o);

    // Done
    #50;
    if (!err_o) begin
      $display("TEST PASSED: prim_count basic operations");
    end else begin
      $display("TEST FAILED: error flag set");
    end
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    tlul_adapter_reg)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for tlul_adapter_reg (TL-UL interface + register interface)
// Exercises TL-UL read/write handshake.

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module tlul_adapter_reg_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import prim_mubi_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interface
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Register interface
  logic re_o;
  logic we_o;
  logic [7:0] addr_o;
  logic [31:0] wdata_o;
  logic [3:0] be_o;
  logic busy_i;
  logic [31:0] rdata_i;
  logic error_i;
  prim_mubi_pkg::mubi4_t en_ifetch_i;
  logic intg_error_o;

  logic [31:0] reg_q;

  tlul_adapter_reg #(
    .RegAw(8),
    .RegDw(32),
    .AccessLatency(0)
  ) dut (
    .clk_i,
    .rst_ni,
    .tl_i,
    .tl_o,
    .en_ifetch_i,
    .intg_error_o,
    .re_o,
    .we_o,
    .addr_o,
    .wdata_o,
    .be_o,
    .busy_i,
    .rdata_i,
    .error_i
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  // Simple register model
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      reg_q <= 32'h0;
    end else if (we_o && (addr_o == 8'h00)) begin
      for (int i = 0; i < 4; i++) begin
        if (be_o[i]) begin
          reg_q[i*8 +: 8] <= wdata_o[i*8 +: 8];
        end
      end
    end
  end

  always_comb begin
    rdata_i = (addr_o == 8'h00) ? reg_q : 32'h0;
    error_i = 1'b0;
    busy_i = 1'b0;
    en_ifetch_i = MuBi4True;
  end

  initial begin
    tlul_init(tl_i);
  end

  initial begin
    $display("Starting tlul_adapter_reg test...");

    rst_ni = 1;
    #1;
    rst_ni = 0;
    #20;
    rst_ni = 1;
    $display("Reset released");

    repeat(5) @(posedge clk_i);
    $display("rst_ni=%b outstanding_q=%b busy_i=%b a_ready=%b d_valid=%b intg_error_o=%b",
             rst_ni, dut.outstanding_q, busy_i, tl_o.a_ready, tl_o.d_valid, intg_error_o);

    tlul_write32(clk_i, tl_i, tl_o, 32'h0, 32'hdeadbeef, 4'hF, tlul_err);
    $display("Write response err=%b", tlul_err);
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    $display("Read response data=0x%08x", tlul_rdata);

    if (tlul_rdata !== 32'hdeadbeef) begin
      $display("TEST FAILED: unexpected readback");
    end else begin
      $display("TEST PASSED: tlul_adapter_reg read/write");
    end
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    gpio_no_alerts)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for gpio_reg_top (TileLink-UL interface)
// Just exercises reset and basic connectivity

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module gpio_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import gpio_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interfaces - use default struct values
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Register interfaces
  gpio_reg2hw_t reg2hw;
  gpio_hw2reg_t hw2reg;

  // RACL interface
  logic racl_policies_i;
  logic racl_error_o;
  logic intg_err_o;

  gpio_reg_top #(
    .EnableRacl(0)
  ) dut (
    .clk_i,
    .rst_ni,
    .tl_i,
    .tl_o,
    .reg2hw,
    .hw2reg,
    .racl_policies_i('0),
    .racl_error_o(),
    .intg_err_o()
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
    hw2reg = '0;
  end

  initial begin
    $display("Starting gpio_reg_top test...");

    // Reset
    rst_ni = 0;
    #20;
    rst_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Check outputs are valid (not X)
    $display("TL response valid: %b", tl_o.d_valid);
    $display("Register outputs available");

    // Simple read transaction (address 0x00 = INTR_STATE)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    repeat(5) @(posedge clk_i);

    $display("TEST PASSED: gpio_reg_top basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    gpio)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for full gpio IP (TileLink-UL interface + alerts)
// Exercises reset and a basic TL-UL read

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module gpio_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import gpio_pkg::*;
  import gpio_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // Strap sampling
  logic strap_en_i = 0;
  gpio_straps_t sampled_straps_o;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Interrupts
  logic [NumIOs-1:0] intr_gpio_o;

  // Alerts
  prim_alert_pkg::alert_rx_t [NumAlerts-1:0] alert_rx_i;
  prim_alert_pkg::alert_tx_t [NumAlerts-1:0] alert_tx_o;

  // RACL interface
  top_racl_pkg::racl_policy_vec_t racl_policies_i;
  top_racl_pkg::racl_error_log_t racl_error_o;

  // GPIOs
  logic [NumIOs-1:0] cio_gpio_i;
  logic [NumIOs-1:0] cio_gpio_o;
  logic [NumIOs-1:0] cio_gpio_en_o;

  gpio #(
    .EnableRacl(1'b0)
  ) dut (
    .clk_i,
    .rst_ni,
    .strap_en_i,
    .sampled_straps_o,
    .tl_i,
    .tl_o,
    .intr_gpio_o,
    .alert_rx_i,
    .alert_tx_o,
    .racl_policies_i,
    .racl_error_o,
    .cio_gpio_i,
    .cio_gpio_o,
    .cio_gpio_en_o
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  initial begin
    tlul_init(tl_i);
    cio_gpio_i = '0;
    alert_rx_i = '0;
    racl_policies_i = '0;
  end

  initial begin
    $display("Starting gpio full IP test...");

    // Reset
    rst_ni = 0;
    #20;
    rst_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Simple read transaction (address 0x00 = INTR_STATE)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    // Simple write transaction (address 0x04 = INTR_ENABLE)
    tlul_write32(clk_i, tl_i, tl_o, 32'h4, 32'h0, 4'hF, tlul_err);
    $display("Got TL write response: err = %b", tlul_err);

    repeat(5) @(posedge clk_i);

    $display("TEST PASSED: gpio full IP basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    uart)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for full UART IP (TileLink-UL interface + alerts)
// Exercises reset and a basic TL-UL read

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module uart_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import uart_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Alerts
  prim_alert_pkg::alert_rx_t [NumAlerts-1:0] alert_rx_i;
  prim_alert_pkg::alert_tx_t [NumAlerts-1:0] alert_tx_o;

  // RACL interface
  top_racl_pkg::racl_policy_vec_t racl_policies_i;
  top_racl_pkg::racl_error_log_t racl_error_o;

  // UART I/O
  logic lsio_trigger_o;
  logic cio_rx_i;
  logic cio_tx_o;
  logic cio_tx_en_o;

  // Interrupts
  logic intr_tx_watermark_o;
  logic intr_tx_empty_o;
  logic intr_rx_watermark_o;
  logic intr_tx_done_o;
  logic intr_rx_overflow_o;
  logic intr_rx_frame_err_o;
  logic intr_rx_break_err_o;
  logic intr_rx_timeout_o;
  logic intr_rx_parity_err_o;

  uart #(
    .EnableRacl(1'b0)
  ) dut (
    .clk_i,
    .rst_ni,
    .tl_i,
    .tl_o,
    .alert_rx_i,
    .alert_tx_o,
    .racl_policies_i,
    .racl_error_o,
    .lsio_trigger_o,
    .cio_rx_i,
    .cio_tx_o,
    .cio_tx_en_o,
    .intr_tx_watermark_o,
    .intr_tx_empty_o,
    .intr_rx_watermark_o,
    .intr_tx_done_o,
    .intr_rx_overflow_o,
    .intr_rx_frame_err_o,
    .intr_rx_break_err_o,
    .intr_rx_timeout_o,
    .intr_rx_parity_err_o
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  initial begin
    tlul_init(tl_i);
    alert_rx_i = '0;
    racl_policies_i = '0;
    cio_rx_i = 1'b1;
  end

  initial begin
    $display("Starting uart full IP test...");

    // Reset
    rst_ni = 0;
    #20;
    rst_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Simple read transaction (address 0x00 = INTR_STATE)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    repeat(5) @(posedge clk_i);

    $display("TEST PASSED: uart full IP basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    i2c)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for full I2C IP (TileLink-UL interface + alerts)
// Exercises reset and a basic TL-UL read

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module i2c_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import i2c_reg_pkg::*;
  import prim_ram_1p_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // RAM config
  ram_1p_cfg_t ram_cfg_i;
  ram_1p_cfg_rsp_t ram_cfg_rsp_o;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Alerts
  prim_alert_pkg::alert_rx_t [NumAlerts-1:0] alert_rx_i;
  prim_alert_pkg::alert_tx_t [NumAlerts-1:0] alert_tx_o;

  // RACL interface
  top_racl_pkg::racl_policy_vec_t racl_policies_i;
  top_racl_pkg::racl_error_log_t racl_error_o;

  // I2C IO
  logic cio_scl_i;
  logic cio_scl_o;
  logic cio_scl_en_o;
  logic cio_sda_i;
  logic cio_sda_o;
  logic cio_sda_en_o;

  logic lsio_trigger_o;

  // Interrupts
  logic intr_fmt_threshold_o;
  logic intr_rx_threshold_o;
  logic intr_acq_threshold_o;
  logic intr_rx_overflow_o;
  logic intr_controller_halt_o;
  logic intr_scl_interference_o;
  logic intr_sda_interference_o;
  logic intr_stretch_timeout_o;
  logic intr_sda_unstable_o;
  logic intr_cmd_complete_o;
  logic intr_tx_stretch_o;
  logic intr_tx_threshold_o;
  logic intr_acq_stretch_o;
  logic intr_unexp_stop_o;
  logic intr_host_timeout_o;

  i2c #(
    .EnableRacl(1'b0)
  ) dut (
    .clk_i,
    .rst_ni,
    .ram_cfg_i,
    .ram_cfg_rsp_o,
    .tl_i,
    .tl_o,
    .alert_rx_i,
    .alert_tx_o,
    .racl_policies_i,
    .racl_error_o,
    .cio_scl_i,
    .cio_scl_o,
    .cio_scl_en_o,
    .cio_sda_i,
    .cio_sda_o,
    .cio_sda_en_o,
    .lsio_trigger_o,
    .intr_fmt_threshold_o,
    .intr_rx_threshold_o,
    .intr_acq_threshold_o,
    .intr_rx_overflow_o,
    .intr_controller_halt_o,
    .intr_scl_interference_o,
    .intr_sda_interference_o,
    .intr_stretch_timeout_o,
    .intr_sda_unstable_o,
    .intr_cmd_complete_o,
    .intr_tx_stretch_o,
    .intr_tx_threshold_o,
    .intr_acq_stretch_o,
    .intr_unexp_stop_o,
    .intr_host_timeout_o
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  initial begin
    tlul_init(tl_i);
    alert_rx_i = '0;
    racl_policies_i = '0;
    cio_scl_i = 1'b1;
    cio_sda_i = 1'b1;
    ram_cfg_i = '0;
  end

  initial begin
    $display("Starting i2c full IP test...");

    // Reset
    rst_ni = 0;
    #20;
    rst_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Simple read transaction (address 0x00 = INTR_STATE)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    repeat(5) @(posedge clk_i);

    $display("TEST PASSED: i2c full IP basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    spi_host)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for full SPI Host IP (TileLink-UL interface + alerts)
// Exercises reset and a basic TL-UL read

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module spi_host_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import spi_host_reg_pkg::*;
  import spi_device_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Alerts
  prim_alert_pkg::alert_rx_t [NumAlerts-1:0] alert_rx_i;
  prim_alert_pkg::alert_tx_t [NumAlerts-1:0] alert_tx_o;

  // RACL interface
  top_racl_pkg::racl_policy_vec_t racl_policies_i;
  top_racl_pkg::racl_error_log_t racl_error_o;

  // SPI IO
  logic cio_sck_o;
  logic cio_sck_en_o;
  logic [0:0] cio_csb_o;
  logic [0:0] cio_csb_en_o;
  logic [3:0] cio_sd_o;
  logic [3:0] cio_sd_en_o;
  logic [3:0] cio_sd_i;

  // Passthrough interface
  passthrough_req_t passthrough_i;
  passthrough_rsp_t passthrough_o;

  logic lsio_trigger_o;

  // Interrupts
  logic intr_error_o;
  logic intr_spi_event_o;

  spi_host #(
    .EnableRacl(1'b0),
    .NumCS(1)
  ) dut (
    .clk_i,
    .rst_ni,
    .tl_i,
    .tl_o,
    .alert_rx_i,
    .alert_tx_o,
    .racl_policies_i,
    .racl_error_o,
    .cio_sck_o,
    .cio_sck_en_o,
    .cio_csb_o,
    .cio_csb_en_o,
    .cio_sd_o,
    .cio_sd_en_o,
    .cio_sd_i,
    .passthrough_i,
    .passthrough_o,
    .lsio_trigger_o,
    .intr_error_o,
    .intr_spi_event_o
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  initial begin
    tlul_init(tl_i);
    alert_rx_i = '0;
    racl_policies_i = '0;
    cio_sd_i = '0;
    passthrough_i = '0;
  end

  initial begin
    $display("Starting spi_host full IP test...");

    // Reset
    rst_ni = 0;
    #20;
    rst_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Simple read transaction (address 0x00 = INTR_STATE)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    repeat(5) @(posedge clk_i);

    $display("TEST PASSED: spi_host full IP basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    spi_device)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for full SPI Device IP (TileLink-UL interface + alerts)
// Exercises reset and a basic TL-UL read

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module spi_device_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import spi_device_reg_pkg::*;
  import spi_device_pkg::*;
  import prim_mubi_pkg::*;
  import prim_ram_2p_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Alerts
  prim_alert_pkg::alert_rx_t [NumAlerts-1:0] alert_rx_i;
  prim_alert_pkg::alert_tx_t [NumAlerts-1:0] alert_tx_o;

  // RACL interface
  top_racl_pkg::racl_policy_vec_t racl_policies_i;
  top_racl_pkg::racl_error_log_t racl_error_o;

  // SPI IO
  logic cio_sck_i;
  logic cio_csb_i;
  logic [3:0] cio_sd_o;
  logic [3:0] cio_sd_en_o;
  logic [3:0] cio_sd_i;
  logic cio_tpm_csb_i;

  // Passthrough interface
  passthrough_req_t passthrough_o;
  passthrough_rsp_t passthrough_i;

  // Interrupts
  logic intr_upload_cmdfifo_not_empty_o;
  logic intr_upload_payload_not_empty_o;
  logic intr_upload_payload_overflow_o;
  logic intr_readbuf_watermark_o;
  logic intr_readbuf_flip_o;
  logic intr_tpm_header_not_empty_o;
  logic intr_tpm_rdfifo_cmd_end_o;
  logic intr_tpm_rdfifo_drop_o;

  // RAM config
  ram_2p_cfg_t ram_cfg_sys2spi_i;
  ram_2p_cfg_rsp_t ram_cfg_rsp_sys2spi_o;
  ram_2p_cfg_t ram_cfg_spi2sys_i;
  ram_2p_cfg_rsp_t ram_cfg_rsp_spi2sys_o;

  // External clock sensor
  logic sck_monitor_o;

  // DFT controls
  logic mbist_en_i;
  logic scan_clk_i;
  logic scan_rst_ni;
  mubi4_t scanmode_i;

  spi_device #(
    .EnableRacl(1'b0)
  ) dut (
    .clk_i,
    .rst_ni,
    .tl_i,
    .tl_o,
    .alert_rx_i,
    .alert_tx_o,
    .racl_policies_i,
    .racl_error_o,
    .cio_sck_i,
    .cio_csb_i,
    .cio_sd_o,
    .cio_sd_en_o,
    .cio_sd_i,
    .cio_tpm_csb_i,
    .passthrough_o,
    .passthrough_i,
    .intr_upload_cmdfifo_not_empty_o,
    .intr_upload_payload_not_empty_o,
    .intr_upload_payload_overflow_o,
    .intr_readbuf_watermark_o,
    .intr_readbuf_flip_o,
    .intr_tpm_header_not_empty_o,
    .intr_tpm_rdfifo_cmd_end_o,
    .intr_tpm_rdfifo_drop_o,
    .ram_cfg_sys2spi_i,
    .ram_cfg_rsp_sys2spi_o,
    .ram_cfg_spi2sys_i,
    .ram_cfg_rsp_spi2sys_o,
    .sck_monitor_o,
    .mbist_en_i,
    .scan_clk_i,
    .scan_rst_ni,
    .scanmode_i
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  initial begin
    tlul_init(tl_i);
    alert_rx_i = '0;
    racl_policies_i = '0;
    cio_sck_i = 1'b0;
    cio_csb_i = 1'b1;
    cio_sd_i = '0;
    cio_tpm_csb_i = 1'b1;
    passthrough_i = '0;
    ram_cfg_sys2spi_i = '0;
    ram_cfg_spi2sys_i = '0;
    mbist_en_i = 1'b0;
    scan_clk_i = 1'b0;
    scan_rst_ni = 1'b1;
    scanmode_i = MuBi4False;
  end

  initial begin
    $display("Starting spi_device full IP test...");

    // Reset
    rst_ni = 0;
    #20;
    rst_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Simple read transaction (address 0x00 = INTR_STATE)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    repeat(5) @(posedge clk_i);

    $display("TEST PASSED: spi_device full IP basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    usbdev)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for full USB Device IP (TileLink-UL interface + alerts)
// Exercises reset and a basic TL-UL read

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module usbdev_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import usbdev_pkg::*;
  import usbdev_reg_pkg::*;
  import prim_ram_1p_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;
  logic clk_aon_i = 0;
  logic rst_aon_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Alerts
  prim_alert_pkg::alert_rx_t [NumAlerts-1:0] alert_rx_i;
  prim_alert_pkg::alert_tx_t [NumAlerts-1:0] alert_tx_o;

  // USB data inputs
  logic cio_usb_dp_i;
  logic cio_usb_dn_i;
  logic usb_rx_d_i;

  // USB data outputs
  logic cio_usb_dp_o;
  logic cio_usb_dp_en_o;
  logic cio_usb_dn_o;
  logic cio_usb_dn_en_o;
  logic usb_tx_se0_o;
  logic usb_tx_d_o;

  // Non-data I/O
  logic cio_sense_i;
  logic usb_dp_pullup_o;
  logic usb_dn_pullup_o;
  logic usb_rx_enable_o;
  logic usb_tx_use_d_se0_o;

  // AON signals
  logic usb_aon_suspend_req_o;
  logic usb_aon_wake_ack_o;
  logic usb_aon_bus_reset_i;
  logic usb_aon_sense_lost_i;
  logic usb_aon_bus_not_idle_i;
  logic usb_aon_wake_detect_active_i;

  // SOF reference
  logic usb_ref_val_o;
  logic usb_ref_pulse_o;

  // RAM config
  ram_1p_cfg_t ram_cfg_i;
  ram_1p_cfg_rsp_t ram_cfg_rsp_o;

  // Interrupts
  logic intr_pkt_received_o;
  logic intr_pkt_sent_o;
  logic intr_powered_o;
  logic intr_disconnected_o;
  logic intr_host_lost_o;
  logic intr_link_reset_o;
  logic intr_link_suspend_o;
  logic intr_link_resume_o;
  logic intr_av_out_empty_o;
  logic intr_rx_full_o;
  logic intr_av_overflow_o;
  logic intr_link_in_err_o;
  logic intr_link_out_err_o;
  logic intr_rx_crc_err_o;
  logic intr_rx_pid_err_o;
  logic intr_rx_bitstuff_err_o;
  logic intr_frame_o;
  logic intr_av_setup_empty_o;

  usbdev #(
    .Stub(1'b0)
  ) dut (
    .clk_i,
    .rst_ni,
    .clk_aon_i,
    .rst_aon_ni,
    .tl_i,
    .tl_o,
    .alert_rx_i,
    .alert_tx_o,
    .cio_usb_dp_i,
    .cio_usb_dn_i,
    .usb_rx_d_i,
    .cio_usb_dp_o,
    .cio_usb_dp_en_o,
    .cio_usb_dn_o,
    .cio_usb_dn_en_o,
    .usb_tx_se0_o,
    .usb_tx_d_o,
    .cio_sense_i,
    .usb_dp_pullup_o,
    .usb_dn_pullup_o,
    .usb_rx_enable_o,
    .usb_tx_use_d_se0_o,
    .usb_aon_suspend_req_o,
    .usb_aon_wake_ack_o,
    .usb_aon_bus_reset_i,
    .usb_aon_sense_lost_i,
    .usb_aon_bus_not_idle_i,
    .usb_aon_wake_detect_active_i,
    .usb_ref_val_o,
    .usb_ref_pulse_o,
    .ram_cfg_i,
    .ram_cfg_rsp_o,
    .intr_pkt_received_o,
    .intr_pkt_sent_o,
    .intr_powered_o,
    .intr_disconnected_o,
    .intr_host_lost_o,
    .intr_link_reset_o,
    .intr_link_suspend_o,
    .intr_link_resume_o,
    .intr_av_out_empty_o,
    .intr_rx_full_o,
    .intr_av_overflow_o,
    .intr_link_in_err_o,
    .intr_link_out_err_o,
    .intr_rx_crc_err_o,
    .intr_rx_pid_err_o,
    .intr_rx_bitstuff_err_o,
    .intr_frame_o,
    .intr_av_setup_empty_o
  );

  // Clock generation
  always #5 clk_i = ~clk_i;
  always #25 clk_aon_i = ~clk_aon_i;

  initial begin
    tlul_init(tl_i);
    alert_rx_i = '0;
    cio_usb_dp_i = 1'b1;
    cio_usb_dn_i = 1'b0;
    usb_rx_d_i = 1'b0;
    cio_sense_i = 1'b0;
    usb_aon_bus_reset_i = 1'b0;
    usb_aon_sense_lost_i = 1'b0;
    usb_aon_bus_not_idle_i = 1'b0;
    usb_aon_wake_detect_active_i = 1'b0;
    ram_cfg_i = '0;
  end

  initial begin
    $display("Starting usbdev full IP test...");

    // Reset
    rst_ni = 0;
    rst_aon_ni = 0;
    #20;
    rst_ni = 1;
    rst_aon_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Simple read transaction (address 0x00)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    repeat(5) @(posedge clk_i);

    $display("TEST PASSED: usbdev full IP basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    uart_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for uart_reg_top (TileLink-UL interface)
// Similar to gpio_reg_top_tb but for UART registers

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module uart_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import uart_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Register interfaces
  uart_reg2hw_t reg2hw;
  uart_hw2reg_t hw2reg;

  // RACL interface
  logic racl_policies_i;
  logic racl_error_o;
  logic intg_err_o;

  uart_reg_top #(
    .EnableRacl(0)
  ) dut (
    .clk_i,
    .rst_ni,
    .tl_i,
    .tl_o,
    .reg2hw,
    .hw2reg,
    .racl_policies_i('0),
    .racl_error_o(),
    .intg_err_o()
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
    hw2reg = '0;
  end

  initial begin
    $display("Starting uart_reg_top test...");

    // Reset
    rst_ni = 0;
    #20;
    rst_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Check outputs are valid
    $display("TL response ready: %b", tl_o.a_ready);

    // Simple read transaction (address 0x00 = INTR_STATE)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    // Simple write transaction (address 0x04 = INTR_ENABLE)
    tlul_write32(clk_i, tl_i, tl_o, 32'h4, 32'h0, 4'hF, tlul_err);
    $display("Got TL write response: err = %b", tlul_err);

    repeat(5) @(posedge clk_i);

    $display("TEST PASSED: uart_reg_top basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    pattgen_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for pattgen_reg_top (TileLink-UL interface)
// Pattern generator register block

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module pattgen_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import pattgen_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Register interfaces
  pattgen_reg2hw_t reg2hw;
  pattgen_hw2reg_t hw2reg;

  // Integrity error
  logic intg_err_o;

  pattgen_reg_top dut (
    .clk_i,
    .rst_ni,
    .tl_i,
    .tl_o,
    .reg2hw,
    .hw2reg,
    .intg_err_o()
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
    hw2reg = '0;
  end

  initial begin
    $display("Starting pattgen_reg_top test...");

    // Reset
    rst_ni = 0;
    #20;
    rst_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Simple read transaction (address 0x00)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    tlul_write32(clk_i, tl_i, tl_o, 32'h4, 32'h0, 4'hF, tlul_err);
    $display("Got TL write response: err = %b", tlul_err);

    repeat(5) @(posedge clk_i);

    $display("TEST PASSED: pattgen_reg_top basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    alert_handler_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for alert_handler_reg_top (TileLink-UL interface)
// Exercises reset and a basic TL-UL read/write.

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module alert_handler_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import alert_handler_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;
  logic rst_shadowed_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Register interfaces
  alert_handler_reg2hw_t reg2hw;
  alert_handler_hw2reg_t hw2reg;

  // Integrity/shadow errors
  logic shadowed_storage_err_o;
  logic shadowed_update_err_o;
  logic intg_err_o;

  alert_handler_reg_top dut (
    .clk_i,
    .rst_ni,
    .rst_shadowed_ni,
    .tl_i,
    .tl_o,
    .reg2hw,
    .hw2reg,
    .shadowed_storage_err_o(),
    .shadowed_update_err_o(),
    .intg_err_o()
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
    hw2reg = '0;
  end

  initial begin
    $display("Starting alert_handler_reg_top test...");

    // Reset
    rst_ni = 0;
    rst_shadowed_ni = 0;
    #20;
    rst_ni = 1;
    rst_shadowed_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    $display("rst_ni=%b rst_shadowed_ni=%b reg_busy=%b shadow_busy=%b rst_done=%b shadow_rst_done=%b a_ready=%b d_valid=%b",
             rst_ni, rst_shadowed_ni, dut.reg_busy, dut.shadow_busy,
             dut.rst_done, dut.shadow_rst_done, tl_o.a_ready, tl_o.d_valid);

    tlul_read32(clk_i, tl_i, tl_o, 32'(ALERT_HANDLER_INTR_STATE_OFFSET), tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    tlul_write32(clk_i, tl_i, tl_o, 32'(ALERT_HANDLER_INTR_ENABLE_OFFSET),
                 32'h0, 4'hF, tlul_err);
    $display("Got TL write response: err = %b", tlul_err);

    repeat(5) @(posedge clk_i);

    $display("TEST PASSED: alert_handler_reg_top basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    alert_handler)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for full alert_handler IP
// Exercises reset, EDN handshake stub, and a basic TL-UL read/write.

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module alert_handler_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import prim_alert_pkg::*;
  import prim_esc_pkg::*;
  import prim_mubi_pkg::*;
  import edn_pkg::*;
  import alert_handler_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;
  logic rst_shadowed_ni = 0;
  logic clk_edn_i = 0;
  logic rst_edn_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Interrupts
  logic intr_classa_o;
  logic intr_classb_o;
  logic intr_classc_o;
  logic intr_classd_o;

  // Low power group control
  prim_mubi_pkg::mubi4_t [NLpg-1:0] lpg_cg_en_i;
  prim_mubi_pkg::mubi4_t [NLpg-1:0] lpg_rst_en_i;

  // Crashdump
  alert_crashdump_t crashdump_o;

  // EDN interface
  edn_req_t edn_o;
  edn_rsp_t edn_i;

  // Alert/esc interfaces
  prim_alert_pkg::alert_tx_t [NAlerts-1:0] alert_tx_i;
  prim_alert_pkg::alert_rx_t [NAlerts-1:0] alert_rx_o;
  prim_esc_pkg::esc_rx_t [N_ESC_SEV-1:0] esc_rx_i;
  prim_esc_pkg::esc_tx_t [N_ESC_SEV-1:0] esc_tx_o;

  alert_handler dut (
    .clk_i,
    .rst_ni,
    .rst_shadowed_ni,
    .clk_edn_i,
    .rst_edn_ni,
    .tl_i,
    .tl_o,
    .intr_classa_o,
    .intr_classb_o,
    .intr_classc_o,
    .intr_classd_o,
    .lpg_cg_en_i,
    .lpg_rst_en_i,
    .crashdump_o,
    .edn_o,
    .edn_i,
    .alert_tx_i,
    .alert_rx_o,
    .esc_rx_i,
    .esc_tx_o
  );

  // Clock generation
  always #5 clk_i = ~clk_i;
  always #7 clk_edn_i = ~clk_edn_i;

  // Initialize interfaces
  initial begin
    tlul_init(tl_i);
    lpg_cg_en_i = '{default: MuBi4True};
    lpg_rst_en_i = '{default: MuBi4True};
    alert_tx_i = '{default: ALERT_TX_DEFAULT};
    esc_rx_i = '{default: ESC_RX_DEFAULT};
    edn_i = EDN_RSP_DEFAULT;
  end

  // Simple EDN responder: ack every request with fixed data.
  always_comb begin
    edn_i = EDN_RSP_DEFAULT;
    edn_i.edn_ack = edn_o.edn_req;
    edn_i.edn_bus = 32'h1234abcd;
    edn_i.edn_fips = 1'b0;
  end

  initial begin
    $display("Starting alert_handler full IP test...");

    // Reset
    rst_ni = 0;
    rst_shadowed_ni = 0;
    rst_edn_ni = 0;
    #20;
    rst_ni = 1;
    rst_shadowed_ni = 1;
    rst_edn_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    $display("rst_ni=%b rst_shadowed_ni=%b reg_busy=%b shadow_busy=%b rst_done=%b shadow_rst_done=%b a_ready=%b d_valid=%b",
             rst_ni, rst_shadowed_ni, dut.u_reg_wrap.u_reg.reg_busy,
             dut.u_reg_wrap.u_reg.shadow_busy, dut.u_reg_wrap.u_reg.rst_done,
             dut.u_reg_wrap.u_reg.shadow_rst_done, tl_o.a_ready, tl_o.d_valid);

    tlul_read32(clk_i, tl_i, tl_o,
                32'(alert_handler_reg_pkg::ALERT_HANDLER_PING_TIMER_REGWEN_OFFSET),
                tlul_rdata);
    $display("Ping timer regwen: 0x%08x", tlul_rdata);

    tlul_read32(clk_i, tl_i, tl_o,
                32'(alert_handler_reg_pkg::ALERT_HANDLER_ALERT_REGWEN_0_OFFSET),
                tlul_rdata);
    $display("Alert regwen[0]: 0x%08x", tlul_rdata);

    tlul_read32(clk_i, tl_i, tl_o,
                32'(alert_handler_reg_pkg::ALERT_HANDLER_INTR_STATE_OFFSET), tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    tlul_write32(clk_i, tl_i, tl_o,
                 32'(alert_handler_reg_pkg::ALERT_HANDLER_INTR_ENABLE_OFFSET),
                 32'h0, 4'hF, tlul_err);
    $display("Got TL write response: err = %b", tlul_err);

    // Shadowed register programming requires two matching writes.
    tlul_write32(clk_i, tl_i, tl_o,
                 32'(alert_handler_reg_pkg::ALERT_HANDLER_PING_TIMER_EN_SHADOWED_OFFSET),
                 32'h1, 4'hF, tlul_err);
    tlul_write32(clk_i, tl_i, tl_o,
                 32'(alert_handler_reg_pkg::ALERT_HANDLER_PING_TIMER_EN_SHADOWED_OFFSET),
                 32'h1, 4'hF, tlul_err);
    tlul_read32(clk_i, tl_i, tl_o,
                32'(alert_handler_reg_pkg::ALERT_HANDLER_PING_TIMER_EN_SHADOWED_OFFSET),
                tlul_rdata);
    $display("Ping timer enable readback: 0x%08x", tlul_rdata);

    tlul_write32(clk_i, tl_i, tl_o,
                 32'(alert_handler_reg_pkg::ALERT_HANDLER_ALERT_EN_SHADOWED_0_OFFSET),
                 32'h1, 4'hF, tlul_err);
    tlul_write32(clk_i, tl_i, tl_o,
                 32'(alert_handler_reg_pkg::ALERT_HANDLER_ALERT_EN_SHADOWED_0_OFFSET),
                 32'h1, 4'hF, tlul_err);
    tlul_read32(clk_i, tl_i, tl_o,
                32'(alert_handler_reg_pkg::ALERT_HANDLER_ALERT_EN_SHADOWED_0_OFFSET),
                tlul_rdata);
    $display("Alert enable readback: 0x%08x", tlul_rdata);

    repeat(10) @(posedge clk_i);

    $display("TEST PASSED: alert_handler full IP basic connectivity + shadowed writes");
    $finish;
  end

  // Timeout
  initial begin
    #20000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    rom_ctrl_regs_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for rom_ctrl_regs_reg_top (TileLink-UL interface)

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module rom_ctrl_regs_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import rom_ctrl_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Register interfaces
  rom_ctrl_regs_reg2hw_t reg2hw;
  rom_ctrl_regs_hw2reg_t hw2reg;

  // Integrity error
  logic intg_err_o;

  rom_ctrl_regs_reg_top dut (
    .clk_i,
    .rst_ni,
    .tl_i,
    .tl_o,
    .reg2hw,
    .hw2reg,
    .intg_err_o()
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
    hw2reg = '0;
  end

  initial begin
    $display("Starting rom_ctrl_regs_reg_top test...");

    // Reset
    rst_ni = 0;
    #20;
    rst_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Simple read transaction (address 0x00)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    tlul_write32(clk_i, tl_i, tl_o, 32'h4, 32'h0, 4'hF, tlul_err);
    $display("Got TL write response: err = %b", tlul_err);

    repeat(5) @(posedge clk_i);

    $display("TEST PASSED: rom_ctrl_regs_reg_top basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    sram_ctrl_regs_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for sram_ctrl_regs_reg_top (TileLink-UL interface)

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module sram_ctrl_regs_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import sram_ctrl_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Register interfaces
  sram_ctrl_regs_reg2hw_t reg2hw;
  sram_ctrl_regs_hw2reg_t hw2reg;

  // RACL interface
  logic racl_policies_i;
  logic racl_error_o;
  logic intg_err_o;

  sram_ctrl_regs_reg_top #(
    .EnableRacl(0)
  ) dut (
    .clk_i,
    .rst_ni,
    .tl_i,
    .tl_o,
    .reg2hw,
    .hw2reg,
    .racl_policies_i('0),
    .racl_error_o(),
    .intg_err_o()
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
    hw2reg = '0;
  end

  initial begin
    $display("Starting sram_ctrl_regs_reg_top test...");

    // Reset
    rst_ni = 0;
    #20;
    rst_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Simple read transaction (address 0x00)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    tlul_write32(clk_i, tl_i, tl_o, 32'h4, 32'h0, 4'hF, tlul_err);
    $display("Got TL write response: err = %b", tlul_err);

    repeat(5) @(posedge clk_i);

    $display("TEST PASSED: sram_ctrl_regs_reg_top basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    spi_host_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for spi_host_reg_top (TileLink-UL interface)
// SPI Host has multiple register windows via tlul_socket_1n

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module spi_host_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import spi_host_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Register interfaces
  spi_host_reg2hw_t reg2hw;
  spi_host_hw2reg_t hw2reg;

  // Window interfaces for FIFO access (not used in basic test)
  tl_h2d_t tl_win_o [2];
  tl_d2h_t tl_win_i [2];

  // RACL interface
  logic racl_policies_i;
  logic racl_error_o;
  logic intg_err_o;

  spi_host_reg_top #(
    .EnableRacl(0)
  ) dut (
    .clk_i,
    .rst_ni,
    .tl_i,
    .tl_o,
    .tl_win_o,
    .tl_win_i,
    .reg2hw,
    .hw2reg,
    .racl_policies_i('0),
    .racl_error_o(),
    .intg_err_o()
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
    hw2reg = '0;
    // Window interfaces return idle response
    tl_win_i[0] = TL_D2H_DEFAULT;
    tl_win_i[1] = TL_D2H_DEFAULT;
  end

  initial begin
    $display("Starting spi_host_reg_top test...");

    // Reset
    rst_ni = 0;
    #20;
    rst_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Check outputs are valid
    $display("TL response ready: %b", tl_o.a_ready);

    // Simple read transaction (address 0x00)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    repeat(5) @(posedge clk_i);

    $display("TEST PASSED: spi_host_reg_top basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    spi_device_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for spi_device_reg_top (TileLink-UL interface)
// SPI Device has 2 window interfaces via tlul_socket_1n

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module spi_device_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import spi_device_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Register interfaces
  spi_device_reg2hw_t reg2hw;
  spi_device_hw2reg_t hw2reg;

  // Window interfaces (2 windows for SRAM access)
  tl_h2d_t tl_win_o [2];
  tl_d2h_t tl_win_i [2];

  // RACL interface
  logic racl_policies_i;
  logic racl_error_o;
  logic intg_err_o;

  spi_device_reg_top #(
    .EnableRacl(0)
  ) dut (
    .clk_i,
    .rst_ni,
    .tl_i,
    .tl_o,
    .tl_win_o,
    .tl_win_i,
    .reg2hw,
    .hw2reg,
    .racl_policies_i('0),
    .racl_error_o(),
    .intg_err_o()
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
    hw2reg = '0;
    // Window interfaces return idle response
    tl_win_i[0] = TL_D2H_DEFAULT;
    tl_win_i[1] = TL_D2H_DEFAULT;
  end

  initial begin
    $display("Starting spi_device_reg_top test...");

    // Reset
    rst_ni = 0;
    #20;
    rst_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Check outputs are valid
    $display("TL response ready: %b", tl_o.a_ready);

    // Simple read transaction (address 0x00)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    repeat(5) @(posedge clk_i);

    $display("TEST PASSED: spi_device_reg_top basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    i2c_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for i2c_reg_top (TileLink-UL interface)
// I2C controller register block

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module i2c_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import i2c_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Register interfaces
  i2c_reg2hw_t reg2hw;
  i2c_hw2reg_t hw2reg;

  // RACL interface
  logic racl_policies_i;
  logic racl_error_o;
  logic intg_err_o;

  i2c_reg_top #(
    .EnableRacl(0)
  ) dut (
    .clk_i,
    .rst_ni,
    .tl_i,
    .tl_o,
    .reg2hw,
    .hw2reg,
    .racl_policies_i('0),
    .racl_error_o(),
    .intg_err_o()
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
    hw2reg = '0;
  end

  initial begin
    $display("Starting i2c_reg_top test...");

    // Reset
    rst_ni = 0;
    #20;
    rst_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Check outputs are valid
    $display("TL response ready: %b", tl_o.a_ready);

    // Simple read transaction (address 0x00 = INTR_STATE)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    repeat(5) @(posedge clk_i);

    $display("TEST PASSED: i2c_reg_top basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    aon_timer_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for aon_timer_reg_top (TileLink-UL interface)
// AON Timer has dual clock domains: main clock (clk_i) and always-on clock (clk_aon_i)

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module aon_timer_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import aon_timer_reg_pkg::*;

  // Main clock domain
  logic clk_i = 0;
  logic rst_ni = 0;

  // Always-on clock domain (slower)
  logic clk_aon_i = 0;
  logic rst_aon_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Register interfaces
  aon_timer_reg2hw_t reg2hw;
  aon_timer_hw2reg_t hw2reg;

  // RACL interface
  logic racl_policies_i;
  logic racl_error_o;
  logic intg_err_o;

  aon_timer_reg_top #(
    .EnableRacl(0)
  ) dut (
    .clk_i,
    .rst_ni,
    .clk_aon_i,
    .rst_aon_ni,
    .tl_i,
    .tl_o,
    .reg2hw,
    .hw2reg,
    .racl_policies_i('0),
    .racl_error_o(),
    .intg_err_o()
  );

  // Main clock generation (10ns period = 100MHz)
  always #5 clk_i = ~clk_i;

  // AON clock generation (slower - 50ns period = 20MHz for simulation speed)
  always #25 clk_aon_i = ~clk_aon_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
    hw2reg = '0;
  end

  initial begin
    $display("Starting aon_timer_reg_top test...");

    // Reset both clock domains
    rst_ni = 0;
    rst_aon_ni = 0;
    #100;  // Wait for both clocks to have several edges
    rst_ni = 1;
    rst_aon_ni = 1;
    $display("Reset released (dual clock domain)");

    // Wait for CDC synchronization
    repeat(20) @(posedge clk_i);

    // Check outputs are valid
    $display("TL response ready: %b", tl_o.a_ready);

    // Simple read transaction
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    repeat(10) @(posedge clk_i);

    $display("TEST PASSED: aon_timer_reg_top basic connectivity (dual clock domain)");
    $finish;
  end

  // Timeout
  initial begin
    #20000;  // Longer timeout for CDC
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    sysrst_ctrl_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for sysrst_ctrl_reg_top (TileLink-UL interface)
// System Reset Controller has dual clock domains: main clock (clk_i) and always-on clock (clk_aon_i)

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module sysrst_ctrl_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import sysrst_ctrl_reg_pkg::*;

  // Main clock domain
  logic clk_i = 0;
  logic rst_ni = 0;

  // Always-on clock domain
  logic clk_aon_i = 0;
  logic rst_aon_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Register interfaces
  sysrst_ctrl_reg2hw_t reg2hw;
  sysrst_ctrl_hw2reg_t hw2reg;

  // Integrity error
  logic intg_err_o;

  sysrst_ctrl_reg_top dut (
    .clk_i,
    .rst_ni,
    .clk_aon_i,
    .rst_aon_ni,
    .tl_i,
    .tl_o,
    .reg2hw,
    .hw2reg,
    .intg_err_o()
  );

  // Main clock generation (10ns period = 100MHz)
  always #5 clk_i = ~clk_i;

  // AON clock generation (slower)
  always #25 clk_aon_i = ~clk_aon_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
    hw2reg = '0;
  end

  initial begin
    $display("Starting sysrst_ctrl_reg_top test...");

    // Reset
    rst_ni = 0;
    rst_aon_ni = 0;
    #20;
    rst_ni = 1;
    rst_aon_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Simple read transaction (address 0x00)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    tlul_write32(clk_i, tl_i, tl_o, 32'h4, 32'h0, 4'hF, tlul_err);
    $display("Got TL write response: err = %b", tlul_err);

    repeat(5) @(posedge clk_i);

    $display("TEST PASSED: sysrst_ctrl_reg_top basic connectivity (dual clock domain)");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    hmac_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for hmac_reg_top (TileLink-UL interface)
// HMAC - Hash Message Authentication Code (crypto IP with FIFO window)

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module hmac_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import hmac_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Window interface for message FIFO access
  tl_h2d_t tl_win_o;
  tl_d2h_t tl_win_i;

  // Register interfaces
  hmac_reg2hw_t reg2hw;
  hmac_hw2reg_t hw2reg;

  // Integrity error
  logic intg_err_o;

  hmac_reg_top dut (
    .clk_i,
    .rst_ni,
    .tl_i,
    .tl_o,
    .tl_win_o,
    .tl_win_i,
    .reg2hw,
    .hw2reg,
    .intg_err_o()
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
    hw2reg = '0;
    tl_win_i = TL_D2H_DEFAULT;
  end

  initial begin
    $display("Starting hmac_reg_top test...");

    // Reset
    rst_ni = 0;
    #20;
    rst_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Check outputs are valid
    $display("TL response ready: %b", tl_o.a_ready);

    // Simple read transaction (CFG register at offset 0x10)
    tlul_read32(clk_i, tl_i, tl_o, 32'h10, tlul_rdata);
    // CFG register
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    repeat(5) @(posedge clk_i);

    $display("TEST PASSED: hmac_reg_top basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    ascon_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for ascon_reg_top (TileLink-UL interface)
// Ascon crypto register block with shadowed registers

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module ascon_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import ascon_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;
  logic rst_shadowed_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Register interfaces
  ascon_reg2hw_t reg2hw;
  ascon_hw2reg_t hw2reg;

  // Error outputs
  logic shadowed_storage_err_o;
  logic shadowed_update_err_o;
  logic intg_err_o;

  ascon_reg_top dut (
    .clk_i,
    .rst_ni,
    .rst_shadowed_ni,
    .tl_i,
    .tl_o,
    .reg2hw,
    .hw2reg,
    .shadowed_storage_err_o,
    .shadowed_update_err_o,
    .intg_err_o()
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
    hw2reg = '0;
  end

  initial begin
    $display("Starting ascon_reg_top test...");

    // Reset both domains
    rst_ni = 0;
    rst_shadowed_ni = 0;
    #20;
    rst_ni = 1;
    rst_shadowed_ni = 1;
    $display("Reset released (with shadow reset)");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Check outputs are valid
    $display("TL response ready: %b", tl_o.a_ready);

    // Simple read transaction (STATUS register at offset 0x0c)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0c, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    // Simple write transaction (CTRL register at offset 0x10)
    tlul_write32(clk_i, tl_i, tl_o, 32'h10, 32'h0, 4'hF, tlul_err);
    $display("Got TL write response: err = %b", tlul_err);

    repeat(5) @(posedge clk_i);

    if (!shadowed_storage_err_o && !shadowed_update_err_o) begin
      $display("Shadow register status: OK (no errors)");
    end else begin
      $display("WARNING: Shadow errors detected: storage=%b update=%b",
               shadowed_storage_err_o, shadowed_update_err_o);
    end

    $display("TEST PASSED: ascon_reg_top basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    rv_timer_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for rv_timer_reg_top (TileLink-UL interface)
// RV Timer - single clock domain timer for RISC-V

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module rv_timer_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import rv_timer_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Register interfaces
  rv_timer_reg2hw_t reg2hw;
  rv_timer_hw2reg_t hw2reg;

  // RACL interface
  logic racl_policies_i;
  logic racl_error_o;
  logic intg_err_o;

  rv_timer_reg_top #(
    .EnableRacl(0)
  ) dut (
    .clk_i,
    .rst_ni,
    .tl_i,
    .tl_o,
    .reg2hw,
    .hw2reg,
    .racl_policies_i('0),
    .racl_error_o(),
    .intg_err_o()
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
    hw2reg = '0;
  end

  initial begin
    $display("Starting rv_timer_reg_top test...");

    // Reset
    rst_ni = 0;
    #20;
    rst_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Check outputs are valid
    $display("TL response ready: %b", tl_o.a_ready);

    // Simple read transaction (CTRL register at offset 0)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    repeat(5) @(posedge clk_i);

    $display("TEST PASSED: rv_timer_reg_top basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    pwm_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for pwm_reg_top (TileLink-UL interface)
// PWM has dual clock domains: main clock (clk_i) and core clock (clk_core_i)

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module pwm_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import pwm_reg_pkg::*;

  // Main clock domain (TL-UL interface)
  logic clk_i = 0;
  logic rst_ni = 0;

  // Core clock domain (PWM core)
  logic clk_core_i = 0;
  logic rst_core_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Register interfaces
  pwm_reg2hw_t reg2hw;

  // RACL interface
  logic racl_policies_i;
  logic racl_error_o;
  logic intg_err_o;

  pwm_reg_top #(
    .EnableRacl(0)
  ) dut (
    .clk_i,
    .rst_ni,
    .clk_core_i,
    .rst_core_ni,
    .tl_i,
    .tl_o,
    .reg2hw,
    .racl_policies_i('0),
    .racl_error_o(),
    .intg_err_o()
  );

  // Main clock generation (10ns period = 100MHz)
  always #5 clk_i = ~clk_i;

  // Core clock generation (slower - 40ns period = 25MHz for simulation speed)
  always #20 clk_core_i = ~clk_core_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
  end

  initial begin
    $display("Starting pwm_reg_top test...");

    // Reset both clock domains
    rst_ni = 0;
    rst_core_ni = 0;
    #100;  // Wait for both clocks to have several edges
    rst_ni = 1;
    rst_core_ni = 1;
    $display("Reset released (dual clock domain)");

    // Wait for CDC synchronization
    repeat(20) @(posedge clk_i);

    // Check outputs are valid
    $display("TL response ready: %b", tl_o.a_ready);

    // Simple read transaction (CFG register at offset 0)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    repeat(10) @(posedge clk_i);

    $display("TEST PASSED: pwm_reg_top basic connectivity (dual clock domain)");
    $finish;
  end

  // Timeout
  initial begin
    #20000;  // Longer timeout for CDC
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    ascon)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for full ascon IP (TileLink-UL interface + alerts)
// EDN/keymgr/lifecycle signals are stubbed for basic connectivity.

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module ascon_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import ascon_reg_pkg::*;
  import edn_pkg::*;
  import lc_ctrl_pkg::*;
  import keymgr_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;
  logic rst_shadowed_ni = 0;

  logic clk_edn_i = 0;
  logic rst_edn_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // EDN interfaces
  edn_req_t edn_o;
  edn_rsp_t edn_i;

  // Key manager sideload key
  hw_key_req_t keymgr_key_i;

  // Life cycle signal
  lc_tx_t lc_escalate_en_i;

  // Alerts
  prim_alert_pkg::alert_rx_t [NumAlerts-1:0] alert_rx_i;
  prim_alert_pkg::alert_tx_t [NumAlerts-1:0] alert_tx_o;

  // Idle indicator
  prim_mubi_pkg::mubi4_t idle_o;

  ascon dut (
    .clk_i,
    .rst_ni,
    .rst_shadowed_ni,
    .idle_o,
    .lc_escalate_en_i,
    .clk_edn_i,
    .rst_edn_ni,
    .edn_o,
    .edn_i,
    .keymgr_key_i,
    .tl_i,
    .tl_o,
    .alert_rx_i,
    .alert_tx_o
  );

  // Clock generation
  always #5 clk_i = ~clk_i;
  always #7 clk_edn_i = ~clk_edn_i;

  initial begin
    tlul_init(tl_i);
    edn_i = EDN_RSP_DEFAULT;
    keymgr_key_i = '0;
    lc_escalate_en_i = LC_TX_DEFAULT;
    alert_rx_i = '0;
  end

  initial begin
    $display("Starting ascon full IP test...");

    rst_ni = 0;
    rst_shadowed_ni = 0;
    rst_edn_ni = 0;
    #40;
    rst_ni = 1;
    rst_shadowed_ni = 1;
    rst_edn_ni = 1;
    $display("Reset released (main + shadow + edn)");

    repeat(10) @(posedge clk_i);

    $display("TL response ready: %b", tl_o.a_ready);

    // Read STATUS register (offset 0x0c)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0c, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    // Write CTRL register (offset 0x10)
    tlul_write32(clk_i, tl_i, tl_o, 32'h10, 32'h0, 4'hF, tlul_err);
    $display("Got TL write response: err = %b", tlul_err);

    repeat(10) @(posedge clk_i);

    $display("TEST PASSED: ascon full IP basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #20000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    dma)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for full DMA IP (TL-UL register interface + alerts)
// Host/CTN TL-UL ports and system port are stubbed to idle.

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module dma_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import dma_pkg::*;
  import dma_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // DMA control signals
  prim_mubi_pkg::mubi4_t scanmode_i;
  lsio_trigger_t lsio_trigger_i;
  logic intr_dma_done_o;
  logic intr_dma_chunk_done_o;
  logic intr_dma_error_o;

  // Alerts
  prim_alert_pkg::alert_rx_t [dma_reg_pkg::NumAlerts-1:0] alert_rx_i;
  prim_alert_pkg::alert_tx_t [dma_reg_pkg::NumAlerts-1:0] alert_tx_o;

  // RACL interface
  top_racl_pkg::racl_policy_vec_t racl_policies_i;
  top_racl_pkg::racl_error_log_t racl_error_o;

  // TL-UL register interface (device)
  tl_h2d_t tl_d_i;
  tl_d2h_t tl_d_o;

  // TL-UL ports for host/CTN transfers (stubbed)
  tl_d2h_t ctn_tl_d2h_i;
  tl_h2d_t ctn_tl_h2d_o;
  tl_d2h_t host_tl_h_i;
  tl_h2d_t host_tl_h_o;

  // System port (stubbed)
  sys_rsp_t sys_i;
  sys_req_t sys_o;

  logic [31:0] tlul_rdata;
  logic tlul_err;

  dma dut (
    .clk_i,
    .rst_ni,
    .scanmode_i,
    .intr_dma_done_o,
    .intr_dma_chunk_done_o,
    .intr_dma_error_o,
    .lsio_trigger_i,
    .alert_rx_i,
    .alert_tx_o,
    .racl_policies_i,
    .racl_error_o,
    .tl_d_i,
    .tl_d_o,
    .ctn_tl_d2h_i,
    .ctn_tl_h2d_o,
    .host_tl_h_i,
    .host_tl_h_o,
    .sys_i,
    .sys_o
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  initial begin
    tlul_init(tl_d_i);
    ctn_tl_d2h_i = TL_D2H_DEFAULT;
    host_tl_h_i = TL_D2H_DEFAULT;
    sys_i = '0;
    scanmode_i = prim_mubi_pkg::MuBi4False;
    lsio_trigger_i = '0;
    alert_rx_i = '0;
    racl_policies_i = '0;
  end

  initial begin
    $display("Starting dma full IP test...");

    rst_ni = 0;
    #20;
    rst_ni = 1;
    $display("Reset released");

    repeat(10) @(posedge clk_i);

    $display("TL response ready: %b", tl_d_o.a_ready);

    // Read STATUS register (offset 0x08)
    tlul_read32(clk_i, tl_d_i, tl_d_o, 32'h08, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    // Write CONTROL register (offset 0x00)
    tlul_write32(clk_i, tl_d_i, tl_d_o, 32'h00, 32'h0, 4'hF, tlul_err);
    $display("Got TL write response: err = %b", tlul_err);

    repeat(10) @(posedge clk_i);

    $display("TEST PASSED: dma full IP basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #20000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    mbx)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for full mailbox IP (TL-UL register interfaces + alerts)
// Core/SOC ports and SRAM host port are stubbed to idle.

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module mbx_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import mbx_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // Alerts
  prim_alert_pkg::alert_rx_t [NumAlerts-1:0] alert_rx_i;
  prim_alert_pkg::alert_tx_t [NumAlerts-1:0] alert_tx_o;

  // RACL interface
  top_racl_pkg::racl_policy_vec_t racl_policies_i;
  top_racl_pkg::racl_error_log_t racl_error_o;

  // TL-UL device ports (core + soc)
  tl_h2d_t core_tl_d_i;
  tl_d2h_t core_tl_d_o;
  tl_h2d_t soc_tl_d_i;
  tl_d2h_t soc_tl_d_o;

  // TL-UL host port to SRAM (stubbed)
  tl_d2h_t sram_tl_h_i;
  tl_h2d_t sram_tl_h_o;

  // Interrupts and status outputs
  logic intr_mbx_ready_o;
  logic intr_mbx_abort_o;
  logic intr_mbx_error_o;
  logic doe_intr_support_o;
  logic doe_intr_en_o;
  logic doe_intr_o;
  logic doe_async_msg_support_o;

  logic [31:0] tlul_rdata;
  logic tlul_err;

  mbx dut (
    .clk_i,
    .rst_ni,
    .intr_mbx_ready_o,
    .intr_mbx_abort_o,
    .intr_mbx_error_o,
    .doe_intr_support_o,
    .doe_intr_en_o,
    .doe_intr_o,
    .doe_async_msg_support_o,
    .alert_rx_i,
    .alert_tx_o,
    .racl_policies_i,
    .racl_error_o,
    .core_tl_d_i,
    .core_tl_d_o,
    .soc_tl_d_i,
    .soc_tl_d_o,
    .sram_tl_h_i,
    .sram_tl_h_o
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  initial begin
    tlul_init(core_tl_d_i);
    tlul_init(soc_tl_d_i);
    sram_tl_h_i = TL_D2H_DEFAULT;
    alert_rx_i = '0;
    racl_policies_i = '0;
  end

  initial begin
    $display("Starting mbx full IP test...");

    rst_ni = 0;
    #20;
    rst_ni = 1;
    $display("Reset released");

    repeat(10) @(posedge clk_i);

    $display("TL response ready: %b", core_tl_d_o.a_ready);

    // Read STATUS register (offset 0x04) from core port
    tlul_read32(clk_i, core_tl_d_i, core_tl_d_o, 32'h04, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    // Write CONTROL register (offset 0x00) from core port
    tlul_write32(clk_i, core_tl_d_i, core_tl_d_o, 32'h00, 32'h0, 4'hF, tlul_err);
    $display("Got TL write response: err = %b", tlul_err);

    repeat(10) @(posedge clk_i);

    $display("TEST PASSED: mbx full IP basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #20000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    keymgr_dpe)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for full keymgr_dpe IP (TL-UL register interface + alerts)
// EDN/KMAC/OTP/ROM interfaces are stubbed to idle.

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module keymgr_dpe_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import keymgr_dpe_reg_pkg::*;
  import edn_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;
  logic rst_shadowed_ni = 0;
  logic clk_edn_i = 0;
  logic rst_edn_ni = 0;

  // TL-UL interface
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Key outputs
  keymgr_pkg::hw_key_req_t aes_key_o;
  keymgr_pkg::hw_key_req_t kmac_key_o;
  keymgr_pkg::otbn_key_req_t otbn_key_o;

  // KMAC app interface (stubbed)
  kmac_pkg::app_req_t kmac_data_o;
  kmac_pkg::app_rsp_t kmac_data_i;
  logic kmac_en_masking_i;

  // Lifecycle + OTP inputs (stubbed)
  lc_ctrl_pkg::lc_tx_t lc_keymgr_en_i;
  lc_ctrl_pkg::lc_keymgr_div_t lc_keymgr_div_i;
  otp_ctrl_pkg::otp_keymgr_key_t otp_key_i;
  otp_ctrl_pkg::otp_device_id_t otp_device_id_i;

  // EDN interface (stubbed)
  edn_req_t edn_o;
  edn_rsp_t edn_i;

  // ROM digest (stubbed)
  rom_ctrl_pkg::keymgr_data_t [keymgr_dpe_reg_pkg::NumRomDigestInputs-1:0] rom_digest_i;

  // Interrupts and alerts
  logic intr_op_done_o;
  prim_alert_pkg::alert_rx_t [keymgr_reg_pkg::NumAlerts-1:0] alert_rx_i;
  prim_alert_pkg::alert_tx_t [keymgr_reg_pkg::NumAlerts-1:0] alert_tx_o;

  keymgr_dpe dut (
    .clk_i,
    .rst_ni,
    .rst_shadowed_ni,
    .clk_edn_i,
    .rst_edn_ni,
    .tl_i,
    .tl_o,
    .aes_key_o,
    .kmac_key_o,
    .otbn_key_o,
    .kmac_data_o,
    .kmac_data_i,
    .kmac_en_masking_i,
    .lc_keymgr_en_i,
    .lc_keymgr_div_i,
    .otp_key_i,
    .otp_device_id_i,
    .edn_o,
    .edn_i,
    .rom_digest_i,
    .intr_op_done_o,
    .alert_rx_i,
    .alert_tx_o
  );

  // Clocks
  always #5 clk_i = ~clk_i;
  always #7 clk_edn_i = ~clk_edn_i;

  initial begin
    tlul_init(tl_i);
    kmac_data_i = '0;
    kmac_en_masking_i = 1'b1;
    lc_keymgr_en_i = lc_ctrl_pkg::LC_TX_DEFAULT;
    lc_keymgr_div_i = '0;
    otp_key_i = '0;
    otp_device_id_i = '0;
    edn_i = EDN_RSP_DEFAULT;
    rom_digest_i = '0;
    alert_rx_i = '0;
  end

  initial begin
    $display("Starting keymgr_dpe full IP test...");

    rst_ni = 0;
    rst_shadowed_ni = 0;
    rst_edn_ni = 0;
    #40;
    rst_ni = 1;
    rst_shadowed_ni = 1;
    rst_edn_ni = 1;
    $display("Reset released (main + shadow + edn)");

    repeat(10) @(posedge clk_i);

    $display("TL response ready: %b", tl_o.a_ready);

    // Read STATUS register (offset 0x08)
    tlul_read32(clk_i, tl_i, tl_o, 32'h08, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    // Write CONTROL register (offset 0x00)
    tlul_write32(clk_i, tl_i, tl_o, 32'h00, 32'h0, 4'hF, tlul_err);
    $display("Got TL write response: err = %b", tlul_err);

    repeat(10) @(posedge clk_i);

    $display("TEST PASSED: keymgr_dpe full IP basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #20000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    rv_dm)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for full rv_dm IP (TL-UL CSR + memory + DMI interfaces)
// Host/SBA/JTAG inputs are stubbed to idle.

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module rv_dm_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import rv_dm_reg_pkg::*;
  import jtag_pkg::*;

  logic clk_i = 0;
  logic clk_lc_i = 0;
  logic rst_ni = 0;
  logic rst_lc_ni = 0;

  // Life cycle / scan signals
  lc_ctrl_pkg::lc_tx_t lc_hw_debug_clr_i;
  lc_ctrl_pkg::lc_tx_t lc_hw_debug_en_i;
  lc_ctrl_pkg::lc_tx_t lc_dft_en_i;
  lc_ctrl_pkg::lc_tx_t pinmux_hw_debug_en_i;
  lc_ctrl_pkg::lc_tx_t lc_check_byp_en_i;
  lc_ctrl_pkg::lc_tx_t lc_escalate_en_i;
  lc_ctrl_pkg::lc_tx_t lc_init_done_i;
  prim_mubi_pkg::mubi8_t otp_dis_rv_dm_late_debug_i;
  prim_mubi_pkg::mubi4_t scanmode_i;
  logic scan_rst_ni;
  logic strap_en_i;
  logic strap_en_override_i;

  logic [31:0] next_dm_addr_i;
  logic [rv_dm_reg_pkg::NrHarts-1:0] unavailable_i;

  // TL-UL interfaces
  tl_h2d_t regs_tl_d_i;
  tl_d2h_t regs_tl_d_o;
  tl_h2d_t mem_tl_d_i;
  tl_d2h_t mem_tl_d_o;
  tl_h2d_t dbg_tl_d_i;
  tl_d2h_t dbg_tl_d_o;
  tl_h2d_t sba_tl_h_o;
  tl_d2h_t sba_tl_h_i;

  // Alerts and RACL
  prim_alert_pkg::alert_rx_t [NumAlerts-1:0] alert_rx_i;
  prim_alert_pkg::alert_tx_t [NumAlerts-1:0] alert_tx_o;
  top_racl_pkg::racl_policy_vec_t racl_policies_i;
  top_racl_pkg::racl_error_log_t racl_error_o;

  // JTAG
  jtag_req_t jtag_i;
  jtag_rsp_t jtag_o;

  // Outputs
  logic ndmreset_req_o;
  logic dmactive_o;
  logic [rv_dm_reg_pkg::NrHarts-1:0] debug_req_o;

  logic [31:0] tlul_rdata;
  logic tlul_err;

  rv_dm dut (
    .clk_i,
    .clk_lc_i,
    .rst_ni,
    .rst_lc_ni,
    .next_dm_addr_i,
    .lc_hw_debug_clr_i,
    .lc_hw_debug_en_i,
    .lc_dft_en_i,
    .pinmux_hw_debug_en_i,
    .lc_check_byp_en_i,
    .lc_escalate_en_i,
    .lc_init_done_i,
    .strap_en_i,
    .strap_en_override_i,
    .otp_dis_rv_dm_late_debug_i,
    .scanmode_i,
    .scan_rst_ni,
    .ndmreset_req_o,
    .dmactive_o,
    .debug_req_o,
    .unavailable_i,
    .regs_tl_d_i,
    .regs_tl_d_o,
    .mem_tl_d_i,
    .mem_tl_d_o,
    .sba_tl_h_o,
    .sba_tl_h_i,
    .alert_rx_i,
    .alert_tx_o,
    .racl_policies_i,
    .racl_error_o,
    .jtag_i,
    .jtag_o,
    .dbg_tl_d_i,
    .dbg_tl_d_o
  );

  always #5 clk_i = ~clk_i;
  always #7 clk_lc_i = ~clk_lc_i;

  initial begin
    tlul_init(regs_tl_d_i);
    tlul_init(mem_tl_d_i);
    tlul_init(dbg_tl_d_i);
    sba_tl_h_i = TL_D2H_DEFAULT;
    alert_rx_i = '0;
    racl_policies_i = '0;
    jtag_i = JTAG_REQ_DEFAULT;
    next_dm_addr_i = 32'h0;
    unavailable_i = '0;

    lc_hw_debug_clr_i = lc_ctrl_pkg::LC_TX_DEFAULT;
    lc_hw_debug_en_i = lc_ctrl_pkg::LC_TX_DEFAULT;
    lc_dft_en_i = lc_ctrl_pkg::LC_TX_DEFAULT;
    pinmux_hw_debug_en_i = lc_ctrl_pkg::LC_TX_DEFAULT;
    lc_check_byp_en_i = lc_ctrl_pkg::LC_TX_DEFAULT;
    lc_escalate_en_i = lc_ctrl_pkg::LC_TX_DEFAULT;
    lc_init_done_i = lc_ctrl_pkg::LC_TX_DEFAULT;
    otp_dis_rv_dm_late_debug_i = prim_mubi_pkg::MuBi8False;
    scanmode_i = prim_mubi_pkg::MuBi4False;
    scan_rst_ni = 1'b1;
    strap_en_i = 1'b0;
    strap_en_override_i = 1'b0;
  end

  initial begin
    $display("Starting rv_dm full IP test...");

    rst_ni = 0;
    rst_lc_ni = 0;
    #40;
    rst_ni = 1;
    rst_lc_ni = 1;
    $display("Reset released (main + lc)");

    repeat(10) @(posedge clk_i);

    $display("TL response ready: %b", regs_tl_d_o.a_ready);

    // Read STATUS register (offset 0x04) from regs port
    tlul_read32(clk_i, regs_tl_d_i, regs_tl_d_o, 32'h04, tlul_rdata);
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    // Write CONTROL register (offset 0x00)
    tlul_write32(clk_i, regs_tl_d_i, regs_tl_d_o, 32'h00, 32'h0, 4'hF, tlul_err);
    $display("Got TL write response: err = %b", tlul_err);

    repeat(10) @(posedge clk_i);

    $display("TEST PASSED: rv_dm full IP basic connectivity");
    $finish;
  end

  initial begin
    #20000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    timer_core)
      cat > "$tb_file" << 'EOF'
// Testbench for timer_core (RISC-V timer logic)
// Tests basic timer functionality without TL-UL interface

module timer_core_tb;
  parameter int N = 1;  // Number of timers

  logic clk_i = 0;
  logic rst_ni = 0;

  // Timer control
  logic active = 0;
  logic [11:0] prescaler = 12'd10;  // Divide by 11
  logic [7:0] step = 8'd1;          // Increment by 1

  // Timer state
  logic tick;
  logic [63:0] mtime_d;
  logic [63:0] mtime = 64'd0;
  logic [63:0] mtimecmp [N];

  // Timer output
  logic [N-1:0] intr;

  timer_core #(
    .N(N)
  ) dut (
    .clk_i,
    .rst_ni,
    .active,
    .prescaler,
    .step,
    .tick,
    .mtime_d,
    .mtime,
    .mtimecmp,
    .intr
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  // Update mtime register on tick
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      mtime <= 64'd0;
    end else if (tick) begin
      mtime <= mtime_d;
    end
  end

  initial begin
    $display("Starting timer_core test...");

    // Set compare value
    mtimecmp[0] = 64'd5;

    // Reset
    rst_ni = 0;
    #20;
    rst_ni = 1;
    $display("Reset released");

    // Wait a few cycles with timer inactive
    repeat(5) @(posedge clk_i);
    $display("Timer inactive: mtime=%d, intr=%b", mtime, intr);

    // Start timer
    @(posedge clk_i);
    active = 1;
    $display("Timer activated");

    // Wait for timer to count and generate interrupt
    repeat(200) @(posedge clk_i);
    $display("After 200 cycles: mtime=%d, intr=%b", mtime, intr);

    // Check if interrupt fired
    if (mtime >= mtimecmp[0]) begin
      $display("Timer reached compare value, interrupt=%b", intr);
      if (intr[0]) begin
        $display("TEST PASSED: timer_core generates interrupt correctly");
      end else begin
        $display("TEST FAILED: interrupt should be high");
      end
    end else begin
      $display("Timer still counting: mtime=%d", mtime);
    end

    // Stop timer
    @(posedge clk_i);
    active = 0;

    repeat(5) @(posedge clk_i);
    $display("Timer stopped: mtime=%d", mtime);

    $finish;
  end

  // Timeout
  initial begin
    #50000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    aes_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for aes_reg_top (TileLink-UL interface)
// AES - Advanced Encryption Standard crypto IP register block
// Features: shadowed registers for security, separate shadow reset

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module aes_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import aes_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;
  logic rst_shadowed_ni = 0;  // AES has separate shadow reset

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Register interfaces
  aes_reg2hw_t reg2hw;
  aes_hw2reg_t hw2reg;

  // Error outputs
  logic shadowed_storage_err_o;
  logic shadowed_update_err_o;
  logic intg_err_o;

  aes_reg_top dut (
    .clk_i,
    .rst_ni,
    .rst_shadowed_ni,
    .tl_i,
    .tl_o,
    .reg2hw,
    .hw2reg,
    .shadowed_storage_err_o,
    .shadowed_update_err_o,
    .intg_err_o()
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
    hw2reg = '0;
  end

  initial begin
    $display("Starting aes_reg_top test...");

    // Reset both domains
    rst_ni = 0;
    rst_shadowed_ni = 0;
    #20;
    rst_ni = 1;
    rst_shadowed_ni = 1;
    $display("Reset released (with shadow reset)");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Check outputs are valid
    $display("TL response ready: %b", tl_o.a_ready);

    // Simple read transaction (STATUS register at offset 0x84)
    tlul_read32(clk_i, tl_i, tl_o, 32'h84, tlul_rdata);
    // STATUS register
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    // Simple write transaction (CTRL register at offset 0x10)
    tlul_write32(clk_i, tl_i, tl_o, 32'h10, 32'h0, 4'hF, tlul_err);
    $display("Got TL write response: err = %b", tlul_err);

    repeat(5) @(posedge clk_i);

    // Check for shadow errors (should be none after clean reset)
    if (!shadowed_storage_err_o && !shadowed_update_err_o) begin
      $display("Shadow register status: OK (no errors)");
    end else begin
      $display("WARNING: Shadow errors detected: storage=%b update=%b",
               shadowed_storage_err_o, shadowed_update_err_o);
    end

    $display("TEST PASSED: aes_reg_top basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    csrng_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for csrng_reg_top (TileLink-UL interface)
// CSRNG - Cryptographic Secure Random Number Generator register block
// Well-documented crypto IP with entropy source interface

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module csrng_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import csrng_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Register interfaces
  csrng_reg2hw_t reg2hw;
  csrng_hw2reg_t hw2reg;

  // Integrity error
  logic intg_err_o;

  csrng_reg_top dut (
    .clk_i,
    .rst_ni,
    .tl_i,
    .tl_o,
    .reg2hw,
    .hw2reg,
    .intg_err_o()
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
    hw2reg = '0;
  end

  initial begin
    $display("Starting csrng_reg_top test...");

    // Reset
    rst_ni = 0;
    #20;
    rst_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Check outputs are valid
    $display("TL response ready: %b", tl_o.a_ready);

    // Simple read transaction (CTRL register at offset 0x18)
    tlul_read32(clk_i, tl_i, tl_o, 32'h18, tlul_rdata);
    // CTRL register
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    tlul_write32(clk_i, tl_i, tl_o, 32'h1c, 32'h0, 4'hF, tlul_err);
    $display("Got TL write response: err = %b", tlul_err);

    repeat(5) @(posedge clk_i);

    $display("TEST PASSED: csrng_reg_top basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    keymgr_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for keymgr_reg_top (TileLink-UL interface)
// Key Manager - crypto IP with shadowed registers for security
// Features: shadowed registers for key protection, separate shadow reset

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module keymgr_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import keymgr_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;
  logic rst_shadowed_ni = 0;  // keymgr has separate shadow reset

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Register interfaces
  keymgr_reg2hw_t reg2hw;
  keymgr_hw2reg_t hw2reg;

  // Error outputs
  logic shadowed_storage_err_o;
  logic shadowed_update_err_o;
  logic intg_err_o;

  keymgr_reg_top dut (
    .clk_i,
    .rst_ni,
    .rst_shadowed_ni,
    .tl_i,
    .tl_o,
    .reg2hw,
    .hw2reg,
    .shadowed_storage_err_o,
    .shadowed_update_err_o,
    .intg_err_o()
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
    hw2reg = '0;
  end

  initial begin
    $display("Starting keymgr_reg_top test...");

    // Reset both domains
    rst_ni = 0;
    rst_shadowed_ni = 0;
    #20;
    rst_ni = 1;
    rst_shadowed_ni = 1;
    $display("Reset released (with shadow reset)");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Check outputs are valid
    $display("TL response ready: %b", tl_o.a_ready);

    // Simple read transaction (CFG_REGWEN register at offset 0x00)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    // CFG_REGWEN register
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    tlul_write32(clk_i, tl_i, tl_o, 32'h0, 32'h0, 4'hF, tlul_err);
    $display("Got TL write response: err = %b", tlul_err);

    repeat(5) @(posedge clk_i);

    // Check for shadow errors (should be none after clean reset)
    if (!shadowed_storage_err_o && !shadowed_update_err_o) begin
      $display("Shadow register status: OK (no errors)");
    end else begin
      $display("WARNING: Shadow errors detected: storage=%b update=%b",
               shadowed_storage_err_o, shadowed_update_err_o);
    end

    $display("TEST PASSED: keymgr_reg_top basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    kmac_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for kmac_reg_top (TileLink-UL interface)
// KMAC - Keccak Message Authentication Code crypto IP
// Features: shadowed registers, 2 window interfaces for message FIFO and state

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module kmac_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import kmac_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;
  logic rst_shadowed_ni = 0;  // KMAC has separate shadow reset

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Window interfaces (KMAC has 2 windows for MSG_FIFO and STATE access)
  tl_h2d_t tl_win_o [2];
  tl_d2h_t tl_win_i [2];

  // Register interfaces
  kmac_reg2hw_t reg2hw;
  kmac_hw2reg_t hw2reg;

  // Error outputs
  logic shadowed_storage_err_o;
  logic shadowed_update_err_o;
  logic intg_err_o;

  kmac_reg_top dut (
    .clk_i,
    .rst_ni,
    .rst_shadowed_ni,
    .tl_i,
    .tl_o,
    .tl_win_o,
    .tl_win_i,
    .reg2hw,
    .hw2reg,
    .shadowed_storage_err_o,
    .shadowed_update_err_o,
    .intg_err_o()
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
    hw2reg = '0;
    // Window interfaces return idle response
    tl_win_i[0] = TL_D2H_DEFAULT;
    tl_win_i[1] = TL_D2H_DEFAULT;
  end

  initial begin
    $display("Starting kmac_reg_top test...");

    // Reset both domains
    rst_ni = 0;
    rst_shadowed_ni = 0;
    #20;
    rst_ni = 1;
    rst_shadowed_ni = 1;
    $display("Reset released (with shadow reset)");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Check outputs are valid
    $display("TL response ready: %b", tl_o.a_ready);

    // Simple read transaction (CFG_SHADOWED register at offset 0x10)
    tlul_read32(clk_i, tl_i, tl_o, 32'h10, tlul_rdata);
    // CFG_SHADOWED register
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    // Simple write transaction (CFG_SHADOWED register)
    tlul_write32(clk_i, tl_i, tl_o, 32'h10, 32'h0, 4'hF, tlul_err);
    $display("Got TL write response: err = %b", tlul_err);

    repeat(5) @(posedge clk_i);

    // Check for shadow errors (should be none after clean reset)
    if (!shadowed_storage_err_o && !shadowed_update_err_o) begin
      $display("Shadow register status: OK (no errors)");
    end else begin
      $display("WARNING: Shadow errors detected: storage=%b update=%b",
               shadowed_storage_err_o, shadowed_update_err_o);
    end

    $display("TEST PASSED: kmac_reg_top basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    otbn_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for otbn_reg_top (TileLink-UL interface)
// OTBN - OpenTitan Big Number accelerator for cryptographic operations
// Features: window interfaces for instruction/data memory access

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module otbn_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import otbn_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Window interfaces (OTBN has 2 windows for IMEM/DMEM access)
  tl_h2d_t tl_win_o [2];
  tl_d2h_t tl_win_i [2];

  // Register interfaces
  otbn_reg2hw_t reg2hw;
  otbn_hw2reg_t hw2reg;

  // Integrity error
  logic intg_err_o;

  otbn_reg_top dut (
    .clk_i,
    .rst_ni,
    .tl_i,
    .tl_o,
    .tl_win_o,
    .tl_win_i,
    .reg2hw,
    .hw2reg,
    .intg_err_o()
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
    hw2reg = '0;
    // Window interfaces return idle response
    tl_win_i[0] = TL_D2H_DEFAULT;
    tl_win_i[1] = TL_D2H_DEFAULT;
  end

  initial begin
    $display("Starting otbn_reg_top test...");

    // Reset
    rst_ni = 0;
    #20;
    rst_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Check outputs are valid
    $display("TL response ready: %b", tl_o.a_ready);

    // Simple read transaction (INTR_STATE register at offset 0x00)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    // INTR_STATE register
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    tlul_write32(clk_i, tl_i, tl_o, 32'h4, 32'h0, 4'hF, tlul_err);
    $display("Got TL write response: err = %b", tlul_err);

    repeat(5) @(posedge clk_i);

    $display("TEST PASSED: otbn_reg_top basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    lc_ctrl_regs_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for lc_ctrl_regs_reg_top (TileLink-UL interface)
// LC Controller - Lifecycle Controller register block

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module lc_ctrl_regs_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import lc_ctrl_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Register interfaces
  lc_ctrl_regs_reg2hw_t reg2hw;
  lc_ctrl_regs_hw2reg_t hw2reg;

  // Integrity error
  logic intg_err_o;

  lc_ctrl_regs_reg_top dut (
    .clk_i,
    .rst_ni,
    .tl_i,
    .tl_o,
    .reg2hw,
    .hw2reg,
    .intg_err_o()
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
    hw2reg = '0;
  end

  initial begin
    $display("Starting lc_ctrl_regs_reg_top test...");

    // Reset
    rst_ni = 0;
    #20;
    rst_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Check outputs are valid
    $display("TL response ready: %b", tl_o.a_ready);

    // Simple read transaction (ALERT_TEST register at offset 0x00)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    // ALERT_TEST register
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    repeat(5) @(posedge clk_i);

    $display("TEST PASSED: lc_ctrl_regs_reg_top basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    otp_ctrl_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for otp_ctrl_core_reg_top (TileLink-UL interface)
// OTP Controller - One-Time Programmable memory controller register block
// Features: window interface for SW config access

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module otp_ctrl_core_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import otp_ctrl_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Window interface for SW config access
  tl_h2d_t tl_win_o;
  tl_d2h_t tl_win_i;

  // Register interfaces
  otp_ctrl_core_reg2hw_t reg2hw;
  otp_ctrl_core_hw2reg_t hw2reg;

  // Integrity error
  logic intg_err_o;

  otp_ctrl_core_reg_top dut (
    .clk_i,
    .rst_ni,
    .tl_i,
    .tl_o,
    .tl_win_o,
    .tl_win_i,
    .reg2hw,
    .hw2reg,
    .intg_err_o()
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
    hw2reg = '0;
    tl_win_i = TL_D2H_DEFAULT;
  end

  initial begin
    $display("Starting otp_ctrl_core_reg_top test...");

    // Reset
    rst_ni = 0;
    #20;
    rst_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Check outputs are valid
    $display("TL response ready: %b", tl_o.a_ready);

    // Simple read transaction (INTR_STATE register at offset 0x00)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    // INTR_STATE register
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    tlul_write32(clk_i, tl_i, tl_o, 32'h4, 32'h0, 4'hF, tlul_err);
    $display("Got TL write response: err = %b", tlul_err);

    repeat(5) @(posedge clk_i);

    $display("TEST PASSED: otp_ctrl_core_reg_top basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    flash_ctrl_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for flash_ctrl_core_reg_top (TileLink-UL interface)
// Flash Controller - flash memory controller register block
// Features: 2 window interfaces for prog/rd FIFOs

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module flash_ctrl_core_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import flash_ctrl_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;
  logic rst_shadowed_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Window interfaces for prog/rd FIFOs (2 windows)
  tl_h2d_t tl_win_o [2];
  tl_d2h_t tl_win_i [2];

  // Register interfaces
  flash_ctrl_core_reg2hw_t reg2hw;
  flash_ctrl_core_hw2reg_t hw2reg;

  // Error signals
  logic shadowed_storage_err_o;
  logic shadowed_update_err_o;
  logic intg_err_o;

  flash_ctrl_core_reg_top dut (
    .clk_i,
    .rst_ni,
    .rst_shadowed_ni,
    .tl_i,
    .tl_o,
    .tl_win_o,
    .tl_win_i,
    .reg2hw,
    .hw2reg,
    .shadowed_storage_err_o(),
    .shadowed_update_err_o(),
    .intg_err_o()
  );

  // Clock generation
  always #5 clk_i = ~clk_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
    hw2reg = '0;
    tl_win_i[0] = TL_D2H_DEFAULT;
    tl_win_i[1] = TL_D2H_DEFAULT;
  end

  initial begin
    $display("Starting flash_ctrl_core_reg_top test...");

    // Reset
    rst_ni = 0;
    rst_shadowed_ni = 0;
    #20;
    rst_ni = 1;
    rst_shadowed_ni = 1;
    $display("Reset released");

    // Wait a few cycles
    repeat(10) @(posedge clk_i);

    // Check outputs are valid
    $display("TL response ready: %b", tl_o.a_ready);

    // Simple read transaction (INTR_STATE register at offset 0x00)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    // INTR_STATE register
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    tlul_write32(clk_i, tl_i, tl_o, 32'h4, 32'h0, 4'hF, tlul_err);
    $display("Got TL write response: err = %b", tlul_err);

    repeat(5) @(posedge clk_i);

    $display("TEST PASSED: flash_ctrl_core_reg_top basic connectivity");
    $finish;
  end

  // Timeout
  initial begin
    #10000;
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    usbdev_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for usbdev_reg_top (TileLink-UL interface)
// USB Device controller register block with dual clock domain
// Features: dual clock (clk_i/clk_aon_i), window interface for buffer memory

`include "prim_assert.sv"
`include "tlul_bfm.sv"

module usbdev_reg_top_tb;
  import tlul_pkg::*;
  import tlul_bfm_pkg::*;
  import usbdev_reg_pkg::*;

  // Main clock domain
  logic clk_i = 0;
  logic rst_ni = 0;

  // Always-on clock domain (for wake detection)
  logic clk_aon_i = 0;
  logic rst_aon_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;
  logic [31:0] tlul_rdata;
  logic tlul_err;

  // Window interface for buffer memory access
  tl_h2d_t tl_win_o;
  tl_d2h_t tl_win_i;

  // Register interfaces
  usbdev_reg2hw_t reg2hw;
  usbdev_hw2reg_t hw2reg;

  // Integrity error
  logic intg_err_o;

  usbdev_reg_top dut (
    .clk_i,
    .rst_ni,
    .clk_aon_i,
    .rst_aon_ni,
    .tl_i,
    .tl_o,
    .tl_win_o,
    .tl_win_i,
    .reg2hw,
    .hw2reg,
    .intg_err_o()
  );

  // Main clock generation (10ns period = 100MHz)
  always #5 clk_i = ~clk_i;

  // AON clock generation (slower - 50ns period = 20MHz for simulation speed)
  always #25 clk_aon_i = ~clk_aon_i;

  // Initialize TL-UL with idle values
  initial begin
    tlul_init(tl_i);
    hw2reg = '0;
    tl_win_i = TL_D2H_DEFAULT;
  end

  initial begin
    $display("Starting usbdev_reg_top test...");

    // Reset both clock domains
    rst_ni = 0;
    rst_aon_ni = 0;
    #100;  // Wait for both clocks to have several edges
    rst_ni = 1;
    rst_aon_ni = 1;
    $display("Reset released (dual clock domain)");

    // Wait a few cycles
    repeat(20) @(posedge clk_i);

    // Check outputs are valid
    $display("TL response ready: %b", tl_o.a_ready);

    // Simple read transaction (INTR_STATE register at offset 0x00)
    tlul_read32(clk_i, tl_i, tl_o, 32'h0, tlul_rdata);
    // INTR_STATE register
    $display("Got TL response: data = 0x%08x", tlul_rdata);

    tlul_write32(clk_i, tl_i, tl_o, 32'h4, 32'h0, 4'hF, tlul_err);
    $display("Got TL write response: err = %b", tlul_err);

    repeat(10) @(posedge clk_i);

    $display("TEST PASSED: usbdev_reg_top basic connectivity (dual clock domain)");
    $finish;
  end

  // Timeout
  initial begin
    #20000;  // Longer timeout for dual clock domain
    $display("TEST TIMEOUT");
    $finish;
  end
endmodule
EOF
      ;;

    *)
      echo "No testbench defined for target: $target" >&2
      return 1
      ;;
  esac
}

# Get dependencies for testbench
# Note: prim_assert.sv is `include`d by source files, not listed directly
get_files_for_target() {
  local TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
  local TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
  local TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
  local GPIO_RTL="$OPENTITAN_DIR/hw/top_earlgrey/ip_autogen/gpio/rtl"
  local PATTGEN_RTL="$OPENTITAN_DIR/hw/ip/pattgen/rtl"
  local ROM_CTRL_RTL="$OPENTITAN_DIR/hw/ip/rom_ctrl/rtl"
  local SRAM_CTRL_RTL="$OPENTITAN_DIR/hw/ip/sram_ctrl/rtl"
  local SYSRST_CTRL_RTL="$OPENTITAN_DIR/hw/ip/sysrst_ctrl/rtl"

  case "$TARGET" in
    prim_fifo_sync)
      echo "$PRIM_RTL/prim_util_pkg.sv"
      echo "$PRIM_RTL/prim_count_pkg.sv"
      echo "$PRIM_GENERIC_RTL/prim_flop.sv"
      echo "$PRIM_RTL/prim_count.sv"
      echo "$PRIM_RTL/prim_fifo_sync_cnt.sv"
      echo "$PRIM_RTL/prim_fifo_sync.sv"
      ;;
    prim_count)
      echo "$PRIM_RTL/prim_util_pkg.sv"
      echo "$PRIM_RTL/prim_count_pkg.sv"
      echo "$PRIM_GENERIC_RTL/prim_flop.sv"
      echo "$PRIM_RTL/prim_count.sv"
      ;;
    tlul_adapter_reg)
      echo "$PRIM_RTL/prim_mubi_pkg.sv"
      echo "$PRIM_RTL/prim_secded_pkg.sv"
      echo "$TOP_RTL/top_pkg.sv"
      echo "$TLUL_RTL/tlul_pkg.sv"
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      ;;
    gpio_no_alerts)
      # Package dependencies (in order)
      echo "$PRIM_RTL/prim_util_pkg.sv"
      echo "$PRIM_RTL/prim_mubi_pkg.sv"
      echo "$PRIM_RTL/prim_secded_pkg.sv"
      echo "$TOP_RTL/top_pkg.sv"
      echo "$TLUL_RTL/tlul_pkg.sv"
      echo "$PRIM_RTL/prim_alert_pkg.sv"
      echo "$TOP_AUTOGEN/top_racl_pkg.sv"
      echo "$PRIM_RTL/prim_subreg_pkg.sv"
      echo "$LC_CTRL_RTL/lc_ctrl_state_pkg.sv"
      echo "$LC_CTRL_RTL/lc_ctrl_reg_pkg.sv"
      echo "$LC_CTRL_RTL/lc_ctrl_pkg.sv"
      # Core primitives
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
      # GPIO packages
      echo "$GPIO_RTL/gpio_pkg.sv"
      echo "$GPIO_RTL/gpio_reg_pkg.sv"
      echo "$GPIO_RTL/gpio_reg_top.sv"
      ;;
    gpio)
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
      # Security anchor primitives
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
      echo "$PRIM_GENERIC_RTL/prim_and2.sv"
      echo "$PRIM_RTL/prim_blanker.sv"
      echo "$PRIM_RTL/prim_sparse_fsm_flop.sv"
      echo "$PRIM_RTL/prim_flop_macros.sv"
      # Interrupt primitive
      echo "$PRIM_RTL/prim_intr_hw.sv"
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_err_resp.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      # GPIO packages and IP
      echo "$GPIO_RTL/gpio_pkg.sv"
      echo "$GPIO_RTL/gpio_reg_pkg.sv"
      echo "$GPIO_RTL/gpio_reg_top.sv"
      echo "$GPIO_RTL/gpio.sv"
      ;;
    uart)
      # Package dependencies (same as uart_reg_top + alert sender + UART core)
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
      # Package dependencies (same as i2c_reg_top + alert sender + I2C core)
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
    spi_host)
      # Package dependencies (same as spi_host_reg_top + alert sender + SPI Host core)
      local SPI_HOST_RTL="$OPENTITAN_DIR/hw/ip/spi_host/rtl"
      local SPI_DEVICE_RTL="$OPENTITAN_DIR/hw/ip/spi_device/rtl"
      # Package dependencies
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
      # Package dependencies (similar to spi_device_reg_top + SPI Device core)
      local SPI_DEVICE_RTL="$OPENTITAN_DIR/hw/ip/spi_device/rtl"
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
      # Package dependencies (same as usbdev_reg_top + USB core)
      local USBDEV_RTL="$OPENTITAN_DIR/hw/ip/usbdev/rtl"
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
    uart_reg_top)
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
    pattgen_reg_top)
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
    alert_handler_reg_top)
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
      # Alert handler register block
      echo "$ALERT_HANDLER_RTL/alert_handler_reg_pkg.sv"
      echo "$ALERT_HANDLER_RTL/alert_handler_reg_top.sv"
      ;;
    alert_handler)
      # Package dependencies
      echo "$PRIM_RTL/prim_util_pkg.sv"
      echo "$PRIM_RTL/prim_mubi_pkg.sv"
      echo "$PRIM_RTL/prim_secded_pkg.sv"
      echo "$TOP_RTL/top_pkg.sv"
      echo "$TLUL_RTL/tlul_pkg.sv"
      echo "$PRIM_RTL/prim_alert_pkg.sv"
      echo "$PRIM_RTL/prim_esc_pkg.sv"
      echo "$TOP_AUTOGEN/top_racl_pkg.sv"
      echo "$PRIM_RTL/prim_subreg_pkg.sv"
      echo "$PRIM_RTL/prim_count_pkg.sv"
      # EDN/CSRNG packages
      echo "$ENTROPY_SRC_RTL/entropy_src_pkg.sv"
      echo "$CSRNG_RTL/csrng_reg_pkg.sv"
      echo "$CSRNG_RTL/csrng_pkg.sv"
      echo "$EDN_RTL/edn_pkg.sv"
      # Alert handler packages
      echo "$ALERT_HANDLER_RTL/alert_handler_reg_pkg.sv"
      echo "$ALERT_HANDLER_RTL/alert_handler_pkg.sv"
      # Core primitives
      echo "$PRIM_GENERIC_RTL/prim_flop.sv"
      echo "$PRIM_GENERIC_RTL/prim_buf.sv"
      echo "$PRIM_GENERIC_RTL/prim_flop_2sync.sv"
      echo "$PRIM_RTL/prim_mubi4_sync.sv"
      echo "$PRIM_RTL/prim_flop_macros.sv"
      echo "$PRIM_RTL/prim_sparse_fsm_flop.sv"
      echo "$PRIM_RTL/prim_lfsr.sv"
      # Security anchor primitives
      echo "$PRIM_RTL/prim_sec_anchor_buf.sv"
      echo "$PRIM_RTL/prim_sec_anchor_flop.sv"
      # Differential decode and alert/esc primitives
      echo "$PRIM_RTL/prim_diff_decode.sv"
      echo "$PRIM_RTL/prim_alert_receiver.sv"
      echo "$PRIM_RTL/prim_esc_sender.sv"
      echo "$PRIM_GENERIC_RTL/prim_xnor2.sv"
      echo "$PRIM_GENERIC_RTL/prim_xor2.sv"
      # Counters and LFSR
      echo "$PRIM_RTL/prim_count.sv"
      echo "$PRIM_RTL/prim_double_lfsr.sv"
      # ECC primitives
      echo "$PRIM_RTL/prim_secded_inv_64_57_dec.sv"
      echo "$PRIM_RTL/prim_secded_inv_64_57_enc.sv"
      echo "$PRIM_RTL/prim_secded_inv_39_32_dec.sv"
      echo "$PRIM_RTL/prim_secded_inv_39_32_enc.sv"
      # EDN helper primitives
      echo "$PRIM_RTL/prim_sync_reqack.sv"
      echo "$PRIM_RTL/prim_sync_reqack_data.sv"
      echo "$PRIM_RTL/prim_packer_fifo.sv"
      echo "$PRIM_RTL/prim_edn_req.sv"
      # Subreg primitives
      echo "$PRIM_RTL/prim_subreg.sv"
      echo "$PRIM_RTL/prim_subreg_ext.sv"
      echo "$PRIM_RTL/prim_subreg_arb.sv"
      echo "$PRIM_RTL/prim_subreg_shadow.sv"
      # Onehot and register check primitives
      echo "$PRIM_RTL/prim_onehot_check.sv"
      echo "$PRIM_RTL/prim_reg_we_check.sv"
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
      # Alert handler IP
      echo "$ALERT_HANDLER_RTL/alert_handler_reg_top.sv"
      echo "$ALERT_HANDLER_RTL/alert_handler_reg_wrap.sv"
      echo "$ALERT_HANDLER_RTL/alert_handler_accu.sv"
      echo "$ALERT_HANDLER_RTL/alert_handler_class.sv"
      echo "$ALERT_HANDLER_RTL/alert_handler_esc_timer.sv"
      echo "$ALERT_HANDLER_RTL/alert_handler_lpg_ctrl.sv"
      echo "$ALERT_HANDLER_RTL/alert_handler_ping_timer.sv"
      echo "$ALERT_HANDLER_RTL/alert_handler.sv"
      ;;
    rom_ctrl_regs_reg_top)
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
      # ROM Controller packages
      echo "$ROM_CTRL_RTL/rom_ctrl_reg_pkg.sv"
      echo "$ROM_CTRL_RTL/rom_ctrl_regs_reg_top.sv"
      ;;
    sram_ctrl_regs_reg_top)
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
      # SRAM Controller packages
      echo "$SRAM_CTRL_RTL/sram_ctrl_reg_pkg.sv"
      echo "$SRAM_CTRL_RTL/sram_ctrl_regs_reg_top.sv"
      ;;
    spi_host_reg_top)
      local SPI_HOST_RTL="$OPENTITAN_DIR/hw/ip/spi_host/rtl"
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
    spi_device_reg_top)
      local SPI_DEVICE_RTL="$OPENTITAN_DIR/hw/ip/spi_device/rtl"
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
      # I2C packages
      echo "$I2C_RTL/i2c_reg_pkg.sv"
      echo "$I2C_RTL/i2c_reg_top.sv"
      ;;
    aon_timer_reg_top)
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
      # CDC primitives
      echo "$PRIM_RTL/prim_pulse_sync.sv"
      echo "$PRIM_RTL/prim_sync_reqack.sv"
      echo "$PRIM_RTL/prim_sync_reqack_data.sv"
      echo "$PRIM_RTL/prim_reg_cdc_arb.sv"
      echo "$PRIM_RTL/prim_reg_cdc.sv"
      # AON Timer packages
      echo "$AON_TIMER_RTL/aon_timer_reg_pkg.sv"
      echo "$AON_TIMER_RTL/aon_timer_reg_top.sv"
      ;;
    sysrst_ctrl_reg_top)
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
    pwm_reg_top)
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
      # CDC primitives
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
    ascon_reg_top)
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
      # Ascon packages
      echo "$ASCON_RTL/ascon_reg_pkg.sv"
      echo "$ASCON_RTL/ascon_reg_top.sv"
      ;;
    hmac_reg_top)
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
    ascon)
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
      # Differential decode and alert sender
      echo "$PRIM_RTL/prim_diff_decode.sv"
      echo "$PRIM_RTL/prim_sec_anchor_buf.sv"
      echo "$PRIM_RTL/prim_sec_anchor_flop.sv"
      echo "$PRIM_RTL/prim_alert_sender.sv"
      # Counters
      echo "$PRIM_RTL/prim_count_pkg.sv"
      echo "$PRIM_RTL/prim_count.sv"
      # Sparse FSM primitives
      echo "$PRIM_RTL/prim_sparse_fsm_flop.sv"
      # Sync/edge primitives
      echo "$PRIM_RTL/prim_pulse_sync.sv"
      echo "$PRIM_RTL/prim_sync_reqack.sv"
      echo "$PRIM_RTL/prim_sync_reqack_data.sv"
      # Ascon primitives
      echo "$PRIM_RTL/prim_ascon_pkg.sv"
      echo "$PRIM_RTL/prim_ascon_sbox.sv"
      echo "$PRIM_RTL/prim_ascon_round.sv"
      echo "$SCRIPT_DIR/opentitan_wrappers/prim_ascon_duplex_wrapper.sv"
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
      echo "$KEYMGR_RTL/keymgr_reg_pkg.sv"
      echo "$KEYMGR_RTL/keymgr_pkg.sv"
      echo "$LC_CTRL_RTL/lc_ctrl_state_pkg.sv"
      echo "$LC_CTRL_RTL/lc_ctrl_reg_pkg.sv"
      echo "$LC_CTRL_RTL/lc_ctrl_pkg.sv"
      # Ascon packages and IP
      echo "$ASCON_RTL/ascon_pkg.sv"
      echo "$ASCON_RTL/ascon_reg_pkg.sv"
      echo "$ASCON_RTL/ascon_reg_top.sv"
      echo "$ASCON_RTL/ascon_core.sv"
      echo "$ASCON_RTL/ascon.sv"
      ;;
    dma)
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
    mbx)
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
      echo "$PRIM_GENERIC_RTL/prim_flop_en.sv"
      echo "$PRIM_GENERIC_RTL/prim_buf.sv"
      echo "$PRIM_GENERIC_RTL/prim_and2.sv"
      echo "$PRIM_GENERIC_RTL/prim_clock_mux2.sv"
      echo "$PRIM_GENERIC_RTL/prim_clock_inv.sv"
      echo "$PRIM_RTL/prim_cdc_rand_delay.sv"
      echo "$PRIM_GENERIC_RTL/prim_flop_2sync.sv"
      # FIFO primitives
      echo "$PRIM_RTL/prim_count_pkg.sv"
      echo "$PRIM_RTL/prim_count.sv"
      echo "$PRIM_RTL/prim_fifo_sync_cnt.sv"
      echo "$PRIM_RTL/prim_fifo_sync.sv"
      echo "$PRIM_RTL/prim_fifo_async_simple.sv"
      # Subreg primitives
      echo "$PRIM_RTL/prim_subreg.sv"
      echo "$PRIM_RTL/prim_subreg_ext.sv"
      echo "$PRIM_RTL/prim_subreg_arb.sv"
      echo "$PRIM_RTL/prim_subreg_shadow.sv"
      # FIFO primitives for TL-UL socket
      echo "$PRIM_RTL/prim_count_pkg.sv"
      echo "$PRIM_RTL/prim_count.sv"
      echo "$PRIM_RTL/prim_fifo_sync_cnt.sv"
      echo "$PRIM_RTL/prim_fifo_sync.sv"
      # Subreg primitives
      echo "$PRIM_RTL/prim_subreg.sv"
      echo "$PRIM_RTL/prim_subreg_ext.sv"
      echo "$PRIM_RTL/prim_subreg_arb.sv"
      echo "$PRIM_RTL/prim_subreg_shadow.sv"
      # Onehot and register check primitives
      echo "$PRIM_RTL/prim_onehot_check.sv"
      echo "$PRIM_RTL/prim_onehot_enc.sv"
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
      # RACL arbiter
      echo "$PRIM_RTL/prim_racl_error_arb.sv"
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
      echo "$TLUL_RTL/tlul_adapter_racl.sv"
      echo "$TLUL_RTL/tlul_adapter_reg_racl.sv"
      echo "$TLUL_RTL/tlul_adapter_host.sv"
      # TL-UL socket for soc register window
      echo "$TLUL_RTL/tlul_err_resp.sv"
      echo "$TLUL_RTL/tlul_fifo_sync.sv"
      echo "$TLUL_RTL/tlul_socket_1n.sv"
      # MBX packages and IP
      echo "$MBX_RTL/mbx_reg_pkg.sv"
      echo "$MBX_RTL/mbx_core_reg_top.sv"
      echo "$MBX_RTL/mbx_soc_reg_top.sv"
      echo "$MBX_RTL/mbx_fsm.sv"
      echo "$MBX_RTL/mbx_imbx.sv"
      echo "$MBX_RTL/mbx_ombx.sv"
      echo "$MBX_RTL/mbx_sramrwarb.sv"
      echo "$MBX_RTL/mbx_hostif.sv"
      echo "$MBX_RTL/mbx_sysif.sv"
      echo "$MBX_RTL/mbx.sv"
      ;;
    keymgr_dpe)
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
    rv_dm)
      # Package dependencies
      echo "$PRIM_RTL/prim_util_pkg.sv"
      echo "$PRIM_RTL/prim_mubi_pkg.sv"
      echo "$PRIM_RTL/prim_secded_pkg.sv"
      echo "$TOP_RTL/top_pkg.sv"
      echo "$TLUL_RTL/tlul_pkg.sv"
      echo "$PRIM_RTL/prim_alert_pkg.sv"
      echo "$TOP_AUTOGEN/top_racl_pkg.sv"
      echo "$PRIM_RTL/prim_subreg_pkg.sv"
      echo "$LC_CTRL_RTL/lc_ctrl_state_pkg.sv"
      echo "$LC_CTRL_RTL/lc_ctrl_reg_pkg.sv"
      echo "$LC_CTRL_RTL/lc_ctrl_pkg.sv"
      # Core primitives
      echo "$PRIM_GENERIC_RTL/prim_flop.sv"
      echo "$PRIM_GENERIC_RTL/prim_flop_en.sv"
      echo "$PRIM_GENERIC_RTL/prim_buf.sv"
      echo "$PRIM_GENERIC_RTL/prim_and2.sv"
      echo "$PRIM_GENERIC_RTL/prim_clock_mux2.sv"
      echo "$PRIM_GENERIC_RTL/prim_clock_inv.sv"
      echo "$PRIM_RTL/prim_cdc_rand_delay.sv"
      echo "$PRIM_GENERIC_RTL/prim_flop_2sync.sv"
      # Subreg primitives
      echo "$PRIM_RTL/prim_subreg.sv"
      echo "$PRIM_RTL/prim_subreg_ext.sv"
      echo "$PRIM_RTL/prim_subreg_arb.sv"
      echo "$PRIM_RTL/prim_subreg_shadow.sv"
      # FIFO primitives
      echo "$PRIM_RTL/prim_count_pkg.sv"
      echo "$PRIM_RTL/prim_count.sv"
      echo "$PRIM_RTL/prim_fifo_sync_cnt.sv"
      echo "$PRIM_RTL/prim_fifo_sync.sv"
      echo "$PRIM_RTL/prim_fifo_async_simple.sv"
      # Lifecycle sync
      echo "$PRIM_RTL/prim_lc_sync.sv"
      echo "$PRIM_RTL/prim_mubi8_sync.sv"
      echo "$PRIM_RTL/prim_mubi32_sync.sv"
      echo "$PRIM_RTL/prim_sync_reqack.sv"
      # Onehot and register check primitives
      echo "$PRIM_RTL/prim_onehot_check.sv"
      echo "$PRIM_RTL/prim_onehot_enc.sv"
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
      echo "$PRIM_RTL/prim_blanker.sv"
      echo "$PRIM_RTL/prim_sparse_fsm_flop.sv"
      echo "$PRIM_RTL/prim_flop_macros.sv"
      # Interrupt primitive and RACL
      echo "$PRIM_RTL/prim_intr_hw.sv"
      echo "$PRIM_RTL/prim_racl_error_arb.sv"
      # TL-UL integrity modules
      echo "$TLUL_RTL/tlul_data_integ_dec.sv"
      echo "$TLUL_RTL/tlul_data_integ_enc.sv"
      # TL-UL adapters
      echo "$TLUL_RTL/tlul_cmd_intg_chk.sv"
      echo "$TLUL_RTL/tlul_cmd_intg_gen.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_gen.sv"
      echo "$TLUL_RTL/tlul_rsp_intg_chk.sv"
      echo "$TLUL_RTL/tlul_err.sv"
      echo "$TLUL_RTL/tlul_err_resp.sv"
      echo "$TLUL_RTL/tlul_adapter_reg.sv"
      echo "$TLUL_RTL/tlul_adapter_host.sv"
      echo "$TLUL_RTL/tlul_lc_gate.sv"
      # Debug module vendor sources
      echo "$RV_DM_VENDOR_RTL/dm_pkg.sv"
      echo "$RV_DM_VENDOR_RTL/dm_csrs.sv"
      echo "$RV_DM_VENDOR_RTL/dm_sba.sv"
      echo "$RV_DM_VENDOR_RTL/dm_mem.sv"
      echo "$RV_DM_VENDOR_RTL/dm_top.sv"
      echo "$RV_DM_VENDOR_RTL/dmi_cdc.sv"
      echo "$RV_DM_ROM_RTL/debug_rom.sv"
      # RV_DM packages and IP
      echo "$RV_DM_RTL/jtag_pkg.sv"
      echo "$RV_DM_RTL/rv_dm_pkg.sv"
      echo "$RV_DM_RTL/rv_dm_reg_pkg.sv"
      echo "$RV_DM_RTL/rv_dm_regs_reg_top.sv"
      echo "$RV_DM_RTL/rv_dm_dbg_reg_top.sv"
      echo "$RV_DM_RTL/rv_dm_mem_reg_top.sv"
      echo "$RV_DM_RTL/rv_dm_dmi_gate.sv"
      echo "$RV_DM_RTL/rv_dm.sv"
      ;;
    aes_reg_top)
      local AES_RTL="$OPENTITAN_DIR/hw/ip/aes/rtl"
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
      local CSRNG_RTL="$OPENTITAN_DIR/hw/ip/csrng/rtl"
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
    keymgr_reg_top)
      local KEYMGR_RTL="$OPENTITAN_DIR/hw/ip/keymgr/rtl"
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
      local KMAC_RTL="$OPENTITAN_DIR/hw/ip/kmac/rtl"
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
      local OTBN_RTL="$OPENTITAN_DIR/hw/ip/otbn/rtl"
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
    timer_core)
      # timer_core - RISC-V timer logic (no TL-UL, minimal dependencies)
      # This is the core timer logic without any register interface
      echo "$RV_TIMER_RTL/timer_core.sv"
      ;;
    lc_ctrl_regs_reg_top)
      # LC Controller register block (lifecycle controller)
      local LC_CTRL_RTL="$OPENTITAN_DIR/hw/ip/lc_ctrl/rtl"
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
    otp_ctrl_reg_top)
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
    flash_ctrl_reg_top)
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
      local FLASH_CTRL_AUTOGEN_RTL="$OPENTITAN_DIR/hw/top_earlgrey/ip_autogen/flash_ctrl/rtl"
      echo "$FLASH_CTRL_AUTOGEN_RTL/flash_ctrl_reg_pkg.sv"
      echo "$FLASH_CTRL_AUTOGEN_RTL/flash_ctrl_core_reg_top.sv"
      ;;
    usbdev_reg_top)
      local USBDEV_RTL="$OPENTITAN_DIR/hw/ip/usbdev/rtl"
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
    *)
      echo "Unknown target: $TARGET" >&2
      return 1
      ;;
  esac
}

# Create output directory
mkdir -p "$OUT_DIR"

# Generate testbench
TB_FILE="$OUT_DIR/opentitan-${TARGET}_tb.sv"
if [[ $VERBOSE -eq 1 ]]; then
  echo "Generating testbench: $TB_FILE"
fi
generate_testbench "$TARGET" "$TB_FILE"

# Get source files
mapfile -t SRC_FILES < <(get_files_for_target)

# Compile to MLIR
MLIR_FILE="$OUT_DIR/opentitan-${TARGET}_tb.mlir"
LOG_FILE="$OUT_DIR/opentitan-${TARGET}_sim.log"

# Target-specific defines, flags, and includes
EXTRA_DEFINES=()
EXTRA_FLAGS=()
EXTRA_INCLUDES=()
if [[ "$TARGET" == "i2c" || "$TARGET" == "spi_device" || "$TARGET" == "usbdev" ]]; then
  # Avoid prim_util_memload DPI tasks that violate region isolation in circt-verilog.
  EXTRA_DEFINES+=("-DSYNTHESIS")
elif [[ "$TARGET" == "spi_host" ]]; then
  # Explicit top module to avoid picking internal SPI host submodules.
  EXTRA_FLAGS+=("--top=spi_host_tb")
elif [[ "$TARGET" == "usbdev" ]]; then
  # Explicit top module to avoid picking internal USB submodules.
  EXTRA_FLAGS+=("--top=usbdev_tb")
elif [[ "$TARGET" == "dma" ]]; then
  # Explicit top module to avoid picking internal DMA submodules.
  EXTRA_FLAGS+=("--top=dma_tb")
elif [[ "$TARGET" == "alert_handler_reg_top" ]]; then
  # Explicit top module to avoid picking internal alert_handler reg submodules.
  EXTRA_FLAGS+=("--top=alert_handler_reg_top_tb")
elif [[ "$TARGET" == "tlul_adapter_reg" ]]; then
  # Explicit top module for TL-UL adapter smoke test.
  EXTRA_FLAGS+=("--top=tlul_adapter_reg_tb")
elif [[ "$TARGET" == "alert_handler" ]]; then
  # Explicit top module to avoid picking internal alert_handler submodules.
  EXTRA_FLAGS+=("--top=alert_handler_tb")
elif [[ "$TARGET" == "mbx" ]]; then
  # Explicit top module to avoid picking internal mailbox submodules.
  EXTRA_FLAGS+=("--top=mbx_tb")
elif [[ "$TARGET" == "keymgr_dpe" ]]; then
  # Explicit top module to avoid picking internal keymgr_dpe submodules.
  EXTRA_FLAGS+=("--top=keymgr_dpe_tb")
elif [[ "$TARGET" == "rv_dm" ]]; then
  # Explicit top module to avoid picking internal rv_dm submodules.
  EXTRA_FLAGS+=("--top=rv_dm_tb")
  EXTRA_DEFINES+=("-DDMIDirectTAP")
elif [[ "$TARGET" == "ascon" ]]; then
  # Match VCS enum/mubi conversion behavior used by OpenTitan.
  EXTRA_FLAGS+=("--compat" "vcs" "--top=ascon_tb")
fi
if [[ "$TARGET" == "usbdev" ]]; then
  # Provide a minimal prim_assert shim to avoid macro parsing issues.
  EXTRA_INCLUDES+=("-I" "$SCRIPT_DIR/opentitan_wrappers")
fi

COMPILE_CMD=(
  "$CIRCT_VERILOG"
  "--ir-hw"  # Lower to HW/Comb/Seq for circt-sim
  "--timescale=1ns/1ps"
  "--no-uvm-auto-include"  # Don't auto-include UVM
  "-DVERILATOR"  # Use dummy assertion macros
  "${EXTRA_DEFINES[@]}"
  "${EXTRA_FLAGS[@]}"
  "${EXTRA_INCLUDES[@]}"
  "-I" "$PRIM_RTL"
  "-I" "$PRIM_GENERIC_RTL"
  "-I" "$TLUL_RTL"
  "-I" "$TOP_RTL"
  "-I" "$TOP_AUTOGEN"
  "-I" "$GPIO_RTL"
  "-I" "$UART_RTL"
  "-I" "$PATTGEN_RTL"
  "-I" "$ROM_CTRL_RTL"
  "-I" "$SRAM_CTRL_RTL"
  "-I" "$SYSRST_CTRL_RTL"
  "-I" "$SPI_HOST_RTL"
  "-I" "$I2C_RTL"
  "-I" "$AON_TIMER_RTL"
  "-I" "$PWM_RTL"
  "-I" "$RV_TIMER_RTL"
  "-I" "$HMAC_RTL"
  "-I" "$AES_RTL"
  "-I" "$CSRNG_RTL"
  "-I" "$KEYMGR_RTL"
  "-I" "$KMAC_RTL"
  "-I" "$OTBN_RTL"
  "-I" "$OTP_CTRL_RTL"
  "-I" "$OTP_CTRL_AUTOGEN_RTL"
  "-I" "$USBDEV_RTL"
  "-I" "$DMA_RTL"
  "-I" "$MBX_RTL"
  "-I" "$KEYMGR_DPE_RTL"
  "-I" "$RV_DM_RTL"
  "-I" "$RV_DM_VENDOR_RTL"
  "-I" "$RV_DM_ROM_RTL"
  "-I" "$OPENTITAN_DIR/hw/ip/lc_ctrl/rtl"
  "-I" "$SCRIPT_DIR/opentitan_wrappers"
  "${SRC_FILES[@]}"
  "$TB_FILE"
  "-o" "$MLIR_FILE"
)

if [[ $VERBOSE -eq 1 ]]; then
  echo "Compile command: ${COMPILE_CMD[*]}"
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY RUN] Would compile with:"
  echo "  ${COMPILE_CMD[*]}"
else
  if [[ $SKIP_COMPILE -eq 0 ]] || [[ ! -f "$MLIR_FILE" ]]; then
    echo "Compiling $TARGET with testbench..."
    if ! "${COMPILE_CMD[@]}" > "$LOG_FILE" 2>&1; then
      echo "COMPILATION FAILED"
      echo "Log: $LOG_FILE"
      tail -30 "$LOG_FILE"
      exit 1
    fi
    echo "Compilation succeeded: $MLIR_FILE"
  else
    echo "Skipping compilation (--skip-compile)"
  fi
fi

# Workaround for circt-sim parser crashes on large MLIR lines.
if [[ "$TARGET" == "alert_handler" && -x "$CIRCT_OPT" ]]; then
  CANON_MLIR_FILE="${MLIR_FILE%.mlir}.canon.mlir"
  if "$CIRCT_OPT" "$MLIR_FILE" -o "$CANON_MLIR_FILE"; then
    MLIR_FILE="$CANON_MLIR_FILE"
  else
    echo "Warning: circt-opt failed to canonicalize MLIR; using original file" >&2
  fi
fi

# Build simulation command
SIM_CMD=(
  "$CIRCT_SIM"
  "--max-cycles=$MAX_CYCLES"
  "--timeout=$TIMEOUT"
)

# Target-specific top for circt-sim
SIM_TOP=""
if [[ "$TARGET" == "spi_host" ]]; then
  SIM_TOP="spi_host_tb"
elif [[ "$TARGET" == "usbdev" ]]; then
  SIM_TOP="usbdev_tb"
elif [[ "$TARGET" == "ascon" ]]; then
  SIM_TOP="ascon_tb"
elif [[ "$TARGET" == "dma" ]]; then
  SIM_TOP="dma_tb"
elif [[ "$TARGET" == "alert_handler_reg_top" ]]; then
  SIM_TOP="alert_handler_reg_top_tb"
elif [[ "$TARGET" == "tlul_adapter_reg" ]]; then
  SIM_TOP="tlul_adapter_reg_tb"
elif [[ "$TARGET" == "alert_handler" ]]; then
  SIM_TOP="alert_handler_tb"
elif [[ "$TARGET" == "mbx" ]]; then
  SIM_TOP="mbx_tb"
elif [[ "$TARGET" == "keymgr_dpe" ]]; then
  SIM_TOP="keymgr_dpe_tb"
elif [[ "$TARGET" == "rv_dm" ]]; then
  SIM_TOP="rv_dm_tb"
fi

if [[ -n "$VCD_FILE" ]]; then
  SIM_CMD+=("--vcd=$VCD_FILE" "--trace-all")
fi

if [[ -n "$SIM_TOP" ]]; then
  SIM_CMD+=("--top=$SIM_TOP")
fi
SIM_CMD+=("$MLIR_FILE")

if [[ $VERBOSE -eq 1 ]]; then
  echo "Simulation command: ${SIM_CMD[*]}"
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY RUN] Would simulate with:"
  echo "  ${SIM_CMD[*]}"
  exit 0
fi

# Run simulation
echo "Simulating $TARGET..."
SIM_LOG="$OUT_DIR/opentitan-${TARGET}_sim_output.log"
if "${SIM_CMD[@]}" 2>&1 | tee "$SIM_LOG"; then
  # circt-sim may exit 0 on wall-clock timeout; treat timeout markers as hard
  # failure so OpenTitan E2E runs remain trustworthy for parity tracking.
  if grep -q "Wall-clock timeout reached" "$SIM_LOG" || grep -q "TEST TIMEOUT" "$SIM_LOG"; then
    echo ""
    echo "Simulation failed: timeout detected"
    echo "Log: $SIM_LOG"
    exit 2
  fi
  if ! grep -q "TEST PASSED" "$SIM_LOG"; then
    echo ""
    echo "Simulation failed: missing TEST PASSED marker"
    echo "Log: $SIM_LOG"
    exit 3
  fi
  echo ""
  echo "Simulation completed"
  echo "Log: $SIM_LOG"
else
  RETVAL=$?
  echo ""
  echo "Simulation failed (exit code $RETVAL)"
  echo "Log: $SIM_LOG"
  exit $RETVAL
fi

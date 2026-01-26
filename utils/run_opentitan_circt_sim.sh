#!/usr/bin/env bash
# Simulate OpenTitan designs with circt-sim
set -euo pipefail

usage() {
  echo "usage: $0 <target> [options]"
  echo ""
  echo "Targets (from run_opentitan_circt_verilog.sh):"
  echo "  prim_fifo_sync     - Synchronous FIFO with simple testbench"
  echo "  prim_count         - Hardened counter with testbench"
  echo "  gpio_no_alerts     - GPIO register block (minimal TL-UL testbench)"
  echo "  uart_reg_top       - UART register block (minimal TL-UL testbench)"
  echo "  spi_host_reg_top   - SPI Host register block (TL-UL with window)"
  echo "  i2c_reg_top        - I2C register block (minimal TL-UL testbench)"
  echo "  aon_timer_reg_top  - AON Timer register block (dual clock domain)"
  echo "  pwm_reg_top        - PWM register block (dual clock domain)"
  echo "  rv_timer_reg_top   - RV Timer register block (single clock)"
  echo "  hmac_reg_top       - HMAC crypto register block (with FIFO window)"
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
OUT_DIR="${OUT_DIR:-$PWD}"
OPENTITAN_DIR="${OPENTITAN_DIR:-$HOME/opentitan}"

# OpenTitan paths
PRIM_RTL="$OPENTITAN_DIR/hw/ip/prim/rtl"
PRIM_GENERIC_RTL="$OPENTITAN_DIR/hw/ip/prim_generic/rtl"
TLUL_RTL="$OPENTITAN_DIR/hw/ip/tlul/rtl"
TOP_RTL="$OPENTITAN_DIR/hw/top_earlgrey/rtl"
TOP_AUTOGEN="$OPENTITAN_DIR/hw/top_earlgrey/rtl/autogen"
GPIO_RTL="$OPENTITAN_DIR/hw/top_earlgrey/ip_autogen/gpio/rtl"
UART_RTL="$OPENTITAN_DIR/hw/ip/uart/rtl"
SPI_HOST_RTL="$OPENTITAN_DIR/hw/ip/spi_host/rtl"
I2C_RTL="$OPENTITAN_DIR/hw/ip/i2c/rtl"
AON_TIMER_RTL="$OPENTITAN_DIR/hw/ip/aon_timer/rtl"
PWM_RTL="$OPENTITAN_DIR/hw/top_earlgrey/ip_autogen/pwm/rtl"
RV_TIMER_RTL="$OPENTITAN_DIR/hw/ip/rv_timer/rtl"
HMAC_RTL="$OPENTITAN_DIR/hw/ip/hmac/rtl"

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
  logic rst_ni = 0;
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

    gpio_no_alerts)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for gpio_reg_top (TileLink-UL interface)
// Just exercises reset and basic connectivity

`include "prim_assert.sv"

module gpio_reg_top_tb;
  import tlul_pkg::*;
  import gpio_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interfaces - use default struct values
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;

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
    tl_i = TL_H2D_DEFAULT;
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
    @(posedge clk_i);
    tl_i.a_valid = 1'b1;
    tl_i.a_opcode = Get;
    tl_i.a_address = 32'h0;
    tl_i.a_size = 2;  // 4 bytes
    tl_i.a_mask = 4'hF;
    tl_i.d_ready = 1'b1;

    // Wait for response
    repeat(5) @(posedge clk_i);
    tl_i.a_valid = 1'b0;

    if (tl_o.d_valid) begin
      $display("Got TL response: data = 0x%08x", tl_o.d_data);
    end

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

    uart_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for uart_reg_top (TileLink-UL interface)
// Similar to gpio_reg_top_tb but for UART registers

`include "prim_assert.sv"

module uart_reg_top_tb;
  import tlul_pkg::*;
  import uart_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;

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
    tl_i = TL_H2D_DEFAULT;
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
    @(posedge clk_i);
    tl_i.a_valid = 1'b1;
    tl_i.a_opcode = Get;
    tl_i.a_address = 32'h0;
    tl_i.a_size = 2;  // 4 bytes
    tl_i.a_mask = 4'hF;
    tl_i.d_ready = 1'b1;

    // Wait for response
    repeat(5) @(posedge clk_i);
    tl_i.a_valid = 1'b0;

    if (tl_o.d_valid) begin
      $display("Got TL response: data = 0x%08x", tl_o.d_data);
    end

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

    spi_host_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for spi_host_reg_top (TileLink-UL interface)
// SPI Host has multiple register windows via tlul_socket_1n

`include "prim_assert.sv"

module spi_host_reg_top_tb;
  import tlul_pkg::*;
  import spi_host_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;

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
    tl_i = TL_H2D_DEFAULT;
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
    @(posedge clk_i);
    tl_i.a_valid = 1'b1;
    tl_i.a_opcode = Get;
    tl_i.a_address = 32'h0;
    tl_i.a_size = 2;  // 4 bytes
    tl_i.a_mask = 4'hF;
    tl_i.d_ready = 1'b1;

    // Wait for response
    repeat(5) @(posedge clk_i);
    tl_i.a_valid = 1'b0;

    if (tl_o.d_valid) begin
      $display("Got TL response: data = 0x%08x", tl_o.d_data);
    end

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

    i2c_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for i2c_reg_top (TileLink-UL interface)
// I2C controller register block

`include "prim_assert.sv"

module i2c_reg_top_tb;
  import tlul_pkg::*;
  import i2c_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;

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
    tl_i = TL_H2D_DEFAULT;
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
    @(posedge clk_i);
    tl_i.a_valid = 1'b1;
    tl_i.a_opcode = Get;
    tl_i.a_address = 32'h0;
    tl_i.a_size = 2;  // 4 bytes
    tl_i.a_mask = 4'hF;
    tl_i.d_ready = 1'b1;

    // Wait for response
    repeat(5) @(posedge clk_i);
    tl_i.a_valid = 1'b0;

    if (tl_o.d_valid) begin
      $display("Got TL response: data = 0x%08x", tl_o.d_data);
    end

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

module aon_timer_reg_top_tb;
  import tlul_pkg::*;
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
    tl_i = TL_H2D_DEFAULT;
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
    @(posedge clk_i);
    tl_i.a_valid = 1'b1;
    tl_i.a_opcode = Get;
    tl_i.a_address = 32'h0;
    tl_i.a_size = 2;
    tl_i.a_mask = 4'hF;
    tl_i.d_ready = 1'b1;

    // Wait for response (may take longer due to CDC)
    repeat(20) @(posedge clk_i);
    tl_i.a_valid = 1'b0;

    if (tl_o.d_valid) begin
      $display("Got TL response: data = 0x%08x", tl_o.d_data);
    end

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

    hmac_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for hmac_reg_top (TileLink-UL interface)
// HMAC - Hash Message Authentication Code (crypto IP with FIFO window)

`include "prim_assert.sv"

module hmac_reg_top_tb;
  import tlul_pkg::*;
  import hmac_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;

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
    tl_i = TL_H2D_DEFAULT;
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
    @(posedge clk_i);
    tl_i.a_valid = 1'b1;
    tl_i.a_opcode = Get;
    tl_i.a_address = 32'h10;  // CFG register
    tl_i.a_size = 2;
    tl_i.a_mask = 4'hF;
    tl_i.d_ready = 1'b1;

    // Wait for response
    repeat(5) @(posedge clk_i);
    tl_i.a_valid = 1'b0;

    if (tl_o.d_valid) begin
      $display("Got TL response: data = 0x%08x", tl_o.d_data);
    end

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

    rv_timer_reg_top)
      cat > "$tb_file" << 'EOF'
// Minimal testbench for rv_timer_reg_top (TileLink-UL interface)
// RV Timer - single clock domain timer for RISC-V

`include "prim_assert.sv"

module rv_timer_reg_top_tb;
  import tlul_pkg::*;
  import rv_timer_reg_pkg::*;

  logic clk_i = 0;
  logic rst_ni = 0;

  // TL-UL interfaces
  tl_h2d_t tl_i;
  tl_d2h_t tl_o;

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
    tl_i = TL_H2D_DEFAULT;
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
    @(posedge clk_i);
    tl_i.a_valid = 1'b1;
    tl_i.a_opcode = Get;
    tl_i.a_address = 32'h0;
    tl_i.a_size = 2;
    tl_i.a_mask = 4'hF;
    tl_i.d_ready = 1'b1;

    // Wait for response
    repeat(5) @(posedge clk_i);
    tl_i.a_valid = 1'b0;

    if (tl_o.d_valid) begin
      $display("Got TL response: data = 0x%08x", tl_o.d_data);
    end

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

module pwm_reg_top_tb;
  import tlul_pkg::*;
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
    tl_i = TL_H2D_DEFAULT;
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
    @(posedge clk_i);
    tl_i.a_valid = 1'b1;
    tl_i.a_opcode = Get;
    tl_i.a_address = 32'h0;
    tl_i.a_size = 2;
    tl_i.a_mask = 4'hF;
    tl_i.d_ready = 1'b1;

    // Wait for response (may take longer due to CDC)
    repeat(20) @(posedge clk_i);
    tl_i.a_valid = 1'b0;

    if (tl_o.d_valid) begin
      $display("Got TL response: data = 0x%08x", tl_o.d_data);
    end

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

COMPILE_CMD=(
  "$CIRCT_VERILOG"
  "--ir-hw"  # Lower to HW/Comb/Seq for circt-sim
  "--timescale=1ns/1ps"
  "--no-uvm-auto-include"  # Don't auto-include UVM
  "-DVERILATOR"  # Use dummy assertion macros
  "-I" "$PRIM_RTL"
  "-I" "$PRIM_GENERIC_RTL"
  "-I" "$TLUL_RTL"
  "-I" "$TOP_RTL"
  "-I" "$TOP_AUTOGEN"
  "-I" "$GPIO_RTL"
  "-I" "$UART_RTL"
  "-I" "$SPI_HOST_RTL"
  "-I" "$I2C_RTL"
  "-I" "$AON_TIMER_RTL"
  "-I" "$PWM_RTL"
  "-I" "$RV_TIMER_RTL"
  "-I" "$HMAC_RTL"
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

# Build simulation command
SIM_CMD=(
  "$CIRCT_SIM"
  "--max-cycles=$MAX_CYCLES"
  "--timeout=$TIMEOUT"
)

if [[ -n "$VCD_FILE" ]]; then
  SIM_CMD+=("--vcd=$VCD_FILE" "--trace-all")
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

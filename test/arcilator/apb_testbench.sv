// RUN: true
// UNSUPPORTED: true
// This test requires AVIP packages to be available.

// Simple APB testbench using the real AVIP interface
import apb_global_pkg::*;

module apb_testbench;
  logic pclk, preset_n;

  // Instantiate the APB interface
  apb_if apb_bus(pclk, preset_n);

  initial begin
    $display("APB Testbench Starting");
    $display("NO_OF_SLAVES = %0d", NO_OF_SLAVES);
    $display("ADDRESS_WIDTH = %0d", ADDRESS_WIDTH);
    $display("DATA_WIDTH = %0d", DATA_WIDTH);
  end
endmodule

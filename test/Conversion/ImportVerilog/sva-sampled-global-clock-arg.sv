// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSampledGlobalClockArg(input logic clk, a);
  global clocking @(posedge clk); endclocking

  // Explicit sampled-value clocking with $global_clock should use the scope
  // global clocking event.
  // CHECK-LABEL: moore.module @SvaSampledGlobalClockArg
  // CHECK: moore.wait_event
  // CHECK: moore.detect_event posedge
  // CHECK: verif.assert
  assert property ($rose(a, @($global_clock)));
endmodule

// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaAssertClockListClockingBlock(input logic clk, req, ack, c);
  clocking cb @(posedge clk); endclocking

  sequence s;
    req ##1 ack;
  endsequence

  // Mixed assertion clock event-lists should resolve clocking-block symbols.
  // CHECK-LABEL: moore.module @SvaAssertClockListClockingBlock
  // CHECK: ltl.matched
  // CHECK: ltl.or
  // CHECK: verif.assert
  assert property (@(s or cb) c);
endmodule

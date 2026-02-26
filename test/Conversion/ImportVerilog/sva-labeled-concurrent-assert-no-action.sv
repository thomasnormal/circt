// RUN: circt-verilog --ir-moore --no-uvm-auto-include %s | FileCheck %s
// REQUIRES: slang

module top(input logic clk, input logic a);
  // Module-level concurrent assertion labels should be preserved on the
  // imported verif.clocked_assert when there is no action block label.
  // CHECK: verif.clocked_assert {{.*}} label "a_must_hold"
  a_must_hold: assert property (@(posedge clk) a);
endmodule

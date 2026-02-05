// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 2 --module=sva_xprop_implication_consequent_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_implication_consequent_sat(input logic clk, input logic in);
  // Consequent has X in next cycle, so implication can be X.
  assert property (@(posedge clk) (1'b1 |-> ##1 in));
endmodule

// CHECK: BMC_RESULT=SAT

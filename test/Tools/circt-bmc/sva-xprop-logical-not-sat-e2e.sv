// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_logical_not_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_logical_not_sat(input logic clk, input logic in);
  // Logical NOT on X should be X, so equality can fail.
  assert property (@(posedge clk) ((!in) == 1'b0));
endmodule

// CHECK: BMC_RESULT=SAT

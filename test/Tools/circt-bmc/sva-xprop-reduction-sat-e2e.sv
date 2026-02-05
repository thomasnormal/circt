// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_reduction_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_reduction_sat(input logic clk, input logic [1:0] in);
  logic red_and;
  logic red_or;
  assign red_and = &in;
  assign red_or = |in;
  // Reduction of X should be X, so equality can fail.
  assert property (@(posedge clk) (red_and == 1'b0));
  assert property (@(posedge clk) (red_or == 1'b0));
endmodule

// CHECK: BMC_RESULT=SAT

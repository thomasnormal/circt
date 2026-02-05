// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_reduction_xor_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_reduction_xor_sat(input logic clk, input logic [1:0] in);
  logic red_xor;
  assign red_xor = ^in;
  // Reduction XOR of X should be X, so equality can fail.
  assert property (@(posedge clk) (red_xor == 1'b0));
endmodule

// CHECK: BMC_RESULT=SAT

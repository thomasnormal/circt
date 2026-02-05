// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_not_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_not_sat(input logic clk, input logic in);
  logic out;
  assign out = ~in;
  // Bitwise NOT should preserve X.
  assert property (@(posedge clk) (out == 1'b0));
endmodule

// CHECK: BMC_RESULT=SAT

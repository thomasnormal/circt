// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 2 --module=sva_xprop_until_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_until_sat(input logic clk, input logic in);
  // Until with X operands can be X, so property can fail.
  assert property (@(posedge clk) (in until 1'b1));
endmodule

// CHECK: BMC_RESULT=SAT

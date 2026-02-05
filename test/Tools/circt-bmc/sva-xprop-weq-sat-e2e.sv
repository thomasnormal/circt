// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_weq_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_weq_sat(input logic clk, input logic [1:0] in);
  // Wildcard equality treats X/Z in rhs as don't-care; lhs X should still yield X.
  assert property (@(posedge clk) ((in ==? 2'b1x) == 1'b0));
endmodule

// CHECK: BMC_RESULT=SAT

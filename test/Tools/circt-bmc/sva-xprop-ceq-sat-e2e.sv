// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_ceq_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_ceq_sat(input logic clk, input logic [1:0] in);
  // Case equality compares X/Z; with unknown inputs, this can be X.
  assert property (@(posedge clk) ((in === 2'b00) == 1'b0));
endmodule

// CHECK: BMC_RESULT=SAT

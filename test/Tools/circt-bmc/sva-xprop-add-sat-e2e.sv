// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_add_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_add_sat(input logic clk, input logic [0:0] in);
  logic [0:0] zero;
  logic [0:0] sum;
  assign zero = 1'b0;
  assign sum = in + zero;
  // sum is X if in is X, so sum == 0 can be X and the assert can fail.
  assert property (@(posedge clk) (sum == 1'b0));
endmodule

// CHECK: BMC_RESULT=SAT

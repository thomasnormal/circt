// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_comb_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_comb_sat(input logic clk, input logic [1:0] in);
  logic [1:0] zero;
  assign zero = 2'b00;
  // in === 2'b00 can be X if in contains unknowns.
  assert property (@(posedge clk) (in === zero) == 1'b0);
endmodule

// CHECK: BMC_RESULT=SAT

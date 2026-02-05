// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_shift_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_shift_sat(input logic clk, input logic [1:0] in);
  logic [1:0] zero;
  logic [1:0] shl;
  assign zero = 2'b00;
  assign shl = in << 1;
  // shl can be X if in has unknown bits, so equality can be X.
  assert property (@(posedge clk) (shl == zero));
endmodule

// CHECK: BMC_RESULT=SAT

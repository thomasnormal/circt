// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_muldiv_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_muldiv_sat(input logic clk, input logic [1:0] in);
  logic [1:0] one;
  logic [1:0] prod;
  logic [1:0] quot;
  assign one = 2'b01;
  assign prod = in * one;
  assign quot = in / one;
  // prod and quot can be X when in is unknown, so equality can fail.
  assert property (@(posedge clk) (prod == 2'b00));
  assert property (@(posedge clk) (quot == 2'b00));
endmodule

// CHECK: BMC_RESULT=SAT

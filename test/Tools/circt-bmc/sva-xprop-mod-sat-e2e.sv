// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_mod_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_mod_sat(input logic clk, input logic [1:0] in);
  logic [1:0] one;
  logic [1:0] rem;
  assign one = 2'b01;
  assign rem = in % one;
  // rem can be X when in is unknown, so equality can fail.
  assert property (@(posedge clk) (rem == 2'b00));
endmodule

// CHECK: BMC_RESULT=SAT

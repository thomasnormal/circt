// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_concat_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_concat_sat(input logic clk, input logic [0:0] in);
  logic [1:0] concat;
  assign concat = {in, 1'b0};
  // Concat should preserve X.
  assert property (@(posedge clk) (concat == 2'b00));
endmodule

// CHECK: BMC_RESULT=SAT

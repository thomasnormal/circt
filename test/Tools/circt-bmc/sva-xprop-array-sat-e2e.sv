// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_array_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_array_sat(input logic clk, input logic [1:0] in);
  logic [1:0] arr [0:0];
  assign arr[0] = in;
  // Array indexing preserves unknowns; equality can be X.
  assert property (@(posedge clk) (arr[0] == 2'b00));
endmodule

// CHECK: BMC_RESULT=SAT

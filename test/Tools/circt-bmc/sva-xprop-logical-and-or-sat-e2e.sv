// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_logical_and_or_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_logical_and_or_sat(input logic clk, input logic in);
  logic and_val;
  logic or_val;
  assign and_val = in && 1'b1;
  assign or_val = in || 1'b0;
  // Logical ops with X should be X, so equality can fail.
  assert property (@(posedge clk) (and_val == 1'b0));
  assert property (@(posedge clk) (or_val == 1'b0));
endmodule

// CHECK: BMC_RESULT=SAT

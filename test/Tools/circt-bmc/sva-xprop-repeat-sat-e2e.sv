// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 3 --module=sva_xprop_repeat_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_repeat_sat(input logic clk, input logic in);
  // Repetition with X operand can be X, so property can fail.
  assert property (@(posedge clk) (in[*2]));
endmodule

// CHECK: BMC_RESULT=SAT

// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc --run-smtlib -b 4 --module=sva_xprop_nonconsecutive_repeat_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: z3

module sva_xprop_nonconsecutive_repeat_sat(input logic clk, input logic in);
  // Non-consecutive repetition with X operand can be X.
  assert property (@(posedge clk) (in[=2]));
endmodule

// CHECK: BMC_RESULT=SAT

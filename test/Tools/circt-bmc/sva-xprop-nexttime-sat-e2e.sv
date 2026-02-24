// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc --run-smtlib -b 2 --module=sva_xprop_nexttime_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: z3

module sva_xprop_nexttime_sat(input logic clk, input logic in);
  // nexttime with X operand can be X, so property can fail.
  assert property (@(posedge clk) nexttime(in));
endmodule

// CHECK: BMC_RESULT=SAT

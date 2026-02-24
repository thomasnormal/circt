// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc --run-smtlib -b 2 --module=sva_xprop_weak_eventually_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: z3

module sva_xprop_weak_eventually_sat(input logic clk, input logic in);
  // Weak eventually with X operand can be X, so property can fail.
  assert property (@(posedge clk) eventually [0:$] in);
endmodule

// CHECK: BMC_RESULT=SAT

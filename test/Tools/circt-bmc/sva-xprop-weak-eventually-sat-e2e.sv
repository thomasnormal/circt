// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 2 --module=sva_xprop_weak_eventually_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_weak_eventually_sat(input logic clk, input logic in);
  // Weak eventually with X operand can be X, so property can fail.
  assert property (@(posedge clk) weak (s_eventually in));
endmodule

// CHECK: BMC_RESULT=SAT

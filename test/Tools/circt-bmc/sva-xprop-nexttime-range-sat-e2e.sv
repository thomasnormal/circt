// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 3 --module=sva_xprop_nexttime_range_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_nexttime_range_sat(input logic clk, input logic in);
  // nexttime range with X operand can be X, so property can fail.
  assert property (@(posedge clk) nexttime[1:2](in));
endmodule

// CHECK: BMC_RESULT=SAT

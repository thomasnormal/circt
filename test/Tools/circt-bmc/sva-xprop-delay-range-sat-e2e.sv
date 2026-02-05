// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 3 --module=sva_xprop_delay_range_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_delay_range_sat(input logic clk, input logic in);
  // Delay range with X operand can be X, so property can fail.
  assert property (@(posedge clk) (in ##[1:2] 1'b1));
endmodule

// CHECK: BMC_RESULT=SAT

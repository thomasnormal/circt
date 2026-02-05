// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 4 --module=sva_xprop_unbounded_delay_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_unbounded_delay_sat(input logic clk, input logic in);
  // Unbounded delay range with X operand can be X.
  assert property (@(posedge clk) (in ##[1:$] 1'b1));
endmodule

// CHECK: BMC_RESULT=SAT

// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_compare_mixed_width_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_compare_mixed_width_sat(input logic clk,
                                         input logic [1:0] in);
  logic [2:0] a;
  assign a = {1'b0, in};
  // Mixed-width comparison can be X when in contains unknowns.
  assert property (@(posedge clk) ((a > 3'b001) == 1'b0));
endmodule

// CHECK: BMC_RESULT=SAT

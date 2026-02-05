// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_compare_signed_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_compare_signed_sat(input logic clk,
                                    input logic signed [1:0] in);
  // Signed comparison can be X when in contains unknowns.
  assert property (@(posedge clk) ((in < -2'sd1) == 1'b0));
endmodule

// CHECK: BMC_RESULT=SAT

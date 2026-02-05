// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_extract_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_extract_sat(input logic clk, input logic [1:0] in);
  logic bit0;
  assign bit0 = in[0];
  // Extraction should preserve X.
  assert property (@(posedge clk) (bit0 == 1'b0));
endmodule

// CHECK: BMC_RESULT=SAT

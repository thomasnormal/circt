// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_signed_shift_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_signed_shift_sat(input logic clk,
                                  input logic signed [1:0] in,
                                  input logic [1:0] shamt);
  logic signed [1:0] out;
  assign out = in >>> shamt;
  // Signed shift with unknowns should be X.
  assert property (@(posedge clk) (out == 2'sb00));
endmodule

// CHECK: BMC_RESULT=SAT

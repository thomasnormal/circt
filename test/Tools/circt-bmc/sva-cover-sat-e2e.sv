// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 2 --module=sva_cover_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_cover_sat(input logic clk);
  logic a;
  assign a = 1'b1;
  cover property (@(posedge clk) a);
endmodule

// CHECK: BMC_RESULT=SAT

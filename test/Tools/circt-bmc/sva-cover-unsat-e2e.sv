// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 2 --module=sva_cover_unsat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_cover_unsat(input logic clk);
  logic a;
  assign a = 1'b0;
  cover property (@(posedge clk) a);
endmodule

// CHECK: BMC_RESULT=UNSAT

// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 2 --module=sva_overlap_implication_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_overlap_implication_sat(input logic clk);
  logic a = 1'b1;
  logic b = 1'b0;
  always_ff @(posedge clk) b <= a;
  assert property (@(posedge clk) a |-> b);
endmodule

// CHECK: BMC_RESULT=SAT

// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 4 --module=sva_nonconsecutive_repeat_concat_unsat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_nonconsecutive_repeat_concat_unsat(input logic clk);
  logic a;
  logic b;
  assign a = 1'b1;
  assign b = 1'b1;
  assert property (@(posedge clk) a [=1:3] ##1 b);
endmodule

// CHECK: BMC_RESULT=UNSAT

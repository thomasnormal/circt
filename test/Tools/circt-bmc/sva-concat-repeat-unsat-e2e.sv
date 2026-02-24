// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc --run-smtlib -b 4 --module=sva_concat_repeat_unsat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: z3

module sva_concat_repeat_unsat(input logic clk);
  logic a;
  logic b;
  assign a = 1'b1;
  assign b = 1'b1;
  assert property (@(posedge clk) a [*2] ##1 b);
endmodule

// CHECK: BMC_RESULT=UNSAT

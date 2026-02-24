// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc --run-smtlib -b 2 --module=sva_nonconsecutive_repeat_unsat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: z3

module sva_nonconsecutive_repeat_unsat(input logic clk);
  logic a;
  assign a = 1'b1;
  assert property (@(posedge clk) a [= 1]);
endmodule

// CHECK: BMC_RESULT=UNSAT

// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc --run-smtlib -b 1 --module=top - | FileCheck %s
// REQUIRES: slang
// REQUIRES: z3

module top;
  logic clk;
  logic a;
  logic b;

  sequence seq;
    @(posedge clk) a ##1 b;
  endsequence

  assert property (seq);
endmodule

// CHECK: Bound reached with no violations!

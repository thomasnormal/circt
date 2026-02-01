// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=top - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
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

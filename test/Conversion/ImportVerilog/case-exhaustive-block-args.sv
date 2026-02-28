// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang
//
// Test for case statement block argument handling when exhaustive case
// branches to a match block that expects block arguments.
// This is a regression test for the "branch has 0 operands for successor #0,
// but target block has 1" error.

// CHECK-LABEL: @testExhaustiveCaseBlockArgs
module testExhaustiveCaseBlockArgs(
  input logic [1:0] sel,
  output logic [3:0] out
);
  // This case statement is exhaustive for 2-bit values.
  // CHECK: cf.cond_br
  // CHECK: cf.cond_br
  // CHECK: cf.br
  always_comb begin
    case (sel)
      2'd0, 2'd1: out = 4'b0001;  // Multiple expressions -> block arg
      2'd2, 2'd3: out = 4'b0010;  // Multiple expressions -> block arg (lastMatchBlock)
    endcase
  end
endmodule

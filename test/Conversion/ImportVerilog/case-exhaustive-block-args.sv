// RUN: circt-verilog --parse-only %s | FileCheck %s
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
  // The issue is that when sel matches multiple expressions in the same
  // case item, a block argument is added to the match block to track
  // which expression matched. When branching to the last match block
  // for the "exhaustive" fallback, we need to provide this argument.
  //
  // CHECK: cf.cond_br {{.*}}, ^[[MATCH0:bb[0-9]+]]
  // CHECK: cf.cond_br {{.*}}, ^[[MATCH0]]
  // CHECK: cf.cond_br {{.*}}, ^[[MATCH1:bb[0-9]+]]
  // CHECK: cf.cond_br {{.*}}, ^[[MATCH1]]
  // For exhaustive case, the fallback branches to last match block with a constant guard
  // CHECK: moore.constant 1
  // CHECK: cf.br ^[[MATCH1]]
  always_comb begin
    case (sel)
      2'd0, 2'd1: out = 4'b0001;  // Multiple expressions -> block arg
      2'd2, 2'd3: out = 4'b0010;  // Multiple expressions -> block arg (lastMatchBlock)
    endcase
  end
endmodule

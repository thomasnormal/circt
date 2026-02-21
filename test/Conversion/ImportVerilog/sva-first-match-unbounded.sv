// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module sva_first_match_unbounded(input logic clk, a, b, c);
  // CHECK-LABEL: moore.module @sva_first_match_unbounded

  // CHECK: ltl.first_match
  assert property (@(posedge clk) first_match(a ##[1:$] b));

  // CHECK: ltl.first_match
  assert property (@(posedge clk) first_match((a [=2]) and (b ##1 c)));

  // CHECK: ltl.first_match
  assert property (@(posedge clk) first_match(@(posedge clk) (a [=2])));
endmodule

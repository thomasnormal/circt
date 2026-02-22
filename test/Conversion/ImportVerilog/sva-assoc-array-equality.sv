// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

module SvaAssocArrayEquality(input logic clk);
  int aa[int], bb[int];

  // Associative-array equality operators should lower through dynamic aggregate
  // comparison rather than scalar bitvector casts.
  // CHECK-LABEL: moore.module @SvaAssocArrayEquality
  // CHECK: moore.array.locator
  // CHECK: ltl.clock
  // CHECK: verif.assert
  assert property (@(posedge clk) aa == bb);
  assert property (@(posedge clk) aa != bb);
  assert property (@(posedge clk) aa === bb);
  assert property (@(posedge clk) aa !== bb);
endmodule

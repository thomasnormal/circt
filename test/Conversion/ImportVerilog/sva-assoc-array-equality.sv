// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

module SvaAssocArrayEquality(input logic clk);
  int aa[int], bb[int];

  // Associative-array equality operators should lower through dynamic aggregate
  // comparison rather than scalar bitvector casts.
  // CHECK-LABEL: moore.module @SvaAssocArrayEquality
  // CHECK: moore.array.locator
  // CHECK: verif.clocked_assert
  assert property (@(posedge clk) aa == bb);
  assert property (@(posedge clk) aa != bb);
  assert property (@(posedge clk) aa === bb);
  assert property (@(posedge clk) aa !== bb);
endmodule

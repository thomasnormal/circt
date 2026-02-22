// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

module SvaAssocArrayEqualityStringKey(input logic clk);
  int aa[string], bb[string];

  // String-key associative array equality should compare both keys and values.
  // CHECK-LABEL: moore.module @SvaAssocArrayEqualityStringKey
  // CHECK: moore.array.locator all, indices
  // CHECK: moore.array.locator all, elements
  // CHECK: verif.assert
  assert property (@(posedge clk) aa == bb);
  assert property (@(posedge clk) aa === bb);
endmodule

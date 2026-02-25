// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

module SvaAssocArrayEqualityStringKey(input logic clk);
  int aa[string], bb[string];

  // String-key associative array equality should compare both keys and values.
  // CHECK-LABEL: moore.module @SvaAssocArrayEqualityStringKey
  // CHECK: moore.array.locator all, indices
  // CHECK: moore.array.locator all, elements
  // CHECK: verif.clocked_assert
  assert property (@(posedge clk) aa == bb);
  assert property (@(posedge clk) aa === bb);
endmodule

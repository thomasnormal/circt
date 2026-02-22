// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

module SvaWildcardAssocArrayEqualityStable(input logic clk, input logic en);
  int aa[*], bb[*];

  // Wildcard associative arrays should lower without verifier failures in
  // equality and sampled-stability contexts.
  // CHECK-LABEL: moore.module @SvaWildcardAssocArrayEqualityStable
  // CHECK: moore.array.locator
  // CHECK: verif.assert
  assert property (@(posedge clk) aa == bb);
  assert property ($stable(aa, @(posedge clk)) |-> en);
endmodule

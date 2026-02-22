// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

module SvaSampledAssocArrayStableStringKeyExplicitClock(
    input logic clk,
    input logic en);
  int aa[string];

  // Typed associative-array sampled stability should compare both keys and
  // values (not only positional value streams).
  // CHECK-LABEL: moore.module @SvaSampledAssocArrayStableStringKeyExplicitClock
  // CHECK: moore.array.locator all, indices
  // CHECK: moore.array.locator all, elements
  // CHECK: verif.assert
  assert property ($stable(aa, @(posedge clk)) |-> en);
endmodule

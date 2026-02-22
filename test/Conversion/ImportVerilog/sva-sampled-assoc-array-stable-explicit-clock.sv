// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

module SvaSampledAssocArrayStableExplicitClock(input logic clk, input logic en);
  int aa[int];

  // Associative arrays should be supported for sampled stability functions
  // with explicit clocking.
  // CHECK-LABEL: moore.module @SvaSampledAssocArrayStableExplicitClock
  // CHECK: moore.procedure always
  // CHECK: moore.wait_event
  // CHECK: verif.assert
  assert property ($stable(aa, @(posedge clk)) |-> en);
endmodule

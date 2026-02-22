// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

module SvaPastAssocArrayExplicitClock(
    input logic clk,
    input logic en,
    input logic [31:0] idx);
  int aa[int];

  // Associative-array values should be supported by sampled-control `$past`
  // helper lowering when explicit clocking and enable are present.
  // CHECK-LABEL: moore.module @SvaPastAssocArrayExplicitClock
  // CHECK: moore.procedure always
  // CHECK: moore.wait_event
  // CHECK: verif.assert
  assert property ($past(aa, 1, en, @(posedge clk))[idx] == aa[idx]);
endmodule

// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSampledStringExplicitClock(input logic clk);
  string s;

  // String sampled-values with explicit clocking should lower via
  // sampled helper-state with string comparison instead of hard failure.
  // CHECK: moore.procedure always
  // CHECK: moore.wait_event
  // CHECK: moore.detect_event posedge
  // CHECK: moore.string_cmp
  // CHECK: verif.assert
  assert property ($changed(s, @(posedge clk)));
endmodule

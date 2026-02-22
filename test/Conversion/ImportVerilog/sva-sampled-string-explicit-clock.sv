// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSampledStringExplicitClock(input logic clk);
  string s;

  // String sampled-values with explicit clocking should lower via
  // string->int sampled helper lowering instead of hard failure.
  // CHECK: moore.procedure always
  // CHECK: moore.wait_event
  // CHECK: moore.detect_event posedge
  // CHECK: moore.string_to_int
  // CHECK: verif.assert
  assert property ($changed(s, @(posedge clk)));
endmodule

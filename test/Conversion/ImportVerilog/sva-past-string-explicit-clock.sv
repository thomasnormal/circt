// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaPastStringExplicitClock(input logic clk);
  string s;

  // String operands in explicit-clocked $past should lower through
  // sampled helper state and string comparison operations.
  // CHECK: moore.procedure always
  // CHECK: moore.wait_event
  // CHECK: moore.detect_event posedge
  // CHECK: moore.string_cmp
  // CHECK: verif.assert
  assert property ($past(s, 1, @(posedge clk)) == s);
endmodule

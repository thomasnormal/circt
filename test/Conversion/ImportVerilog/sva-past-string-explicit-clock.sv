// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaPastStringExplicitClock(input logic clk);
  string s;

  // String operands in explicit-clocked $past should lower through
  // string<->int conversion around sampled helper state.
  // CHECK: moore.procedure always
  // CHECK: moore.wait_event
  // CHECK: moore.detect_event posedge
  // CHECK: moore.string_to_int
  // CHECK: moore.int_to_string
  // CHECK: verif.assert
  assert property ($past(s, 1, @(posedge clk)) == s);
endmodule

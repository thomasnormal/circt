// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaPastUnpackedExplicitClock(input logic clk_a, input logic clk_b);
  logic [1:0] s [2];

  // CHECK: moore.variable : <uarray<2 x l2>>
  // CHECK: moore.procedure always
  // CHECK: moore.blocking_assign
  // CHECK: moore.uarray_cmp eq
  // CHECK: verif.assert
  assert property (@(posedge clk_a) ($past(s, 1, @(posedge clk_b)) == s));
endmodule

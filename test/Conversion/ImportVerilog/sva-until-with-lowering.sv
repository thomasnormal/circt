// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module top(input logic clk, a, b);
  // Weak `until_with` lowers as `lhs until (lhs and rhs)`.
  // CHECK-LABEL: moore.module @top
  // CHECK: [[AND:%.*]] = comb.and bin
  // CHECK: [[UNTIL:%.*]] = ltl.until
  // CHECK-SAME: [[AND]]
  // CHECK: verif.clocked_assert [[UNTIL]], posedge
  assert property (@(posedge clk) a until_with b);
endmodule

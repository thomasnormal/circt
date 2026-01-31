// RUN: circt-verilog --ir-moore --no-uvm-auto-include %s 2>&1 | FileCheck %s
// REQUIRES: slang

// Test that $rose can be used as implication antecedent.
// With the moore.past-based implementation, $rose returns a value type
// that can be properly used in overlapped implication antecedent.

module PropertyTypeValid(input logic clk, a, b);

  // Test: $rose as overlapped implication antecedent
  // $rose returns a moore value type (l1) which can be used as antecedent.
  // CHECK: moore.past
  // CHECK: moore.not
  // CHECK: moore.and
  // CHECK: ltl.implication
  // CHECK: verif.clocked_assert
  assert property (@(posedge clk) $rose(a) |-> b);

endmodule

// RUN: circt-verilog --ir-moore --no-uvm-auto-include %s 2>&1 | FileCheck %s
// REQUIRES: slang

// Test that property types used in invalid contexts produce clear error messages
// instead of crashes or MLIR verifier failures.

module PropertyTypeErrors(input logic clk, a, b);

  // Test: $rose as overlapped implication antecedent
  // $rose returns a property type (!ltl.property) because it uses ltl.and/ltl.not,
  // but implication antecedent requires a sequence type (i1 or !ltl.sequence).
  // CHECK: error: property type cannot be used as implication antecedent; consider restructuring the assertion to use the property as a consequent
  assert property (@(posedge clk) $rose(a) |-> b);

endmodule

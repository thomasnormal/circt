// RUN: circt-verilog --ir-moore --no-uvm-auto-include %s 2>&1 | FileCheck %s
// REQUIRES: slang

// Test that property types in sequence concatenation produce clear error messages.
// This used to cause an assertion failure/crash.

module PropertyConcatError(input logic clk, a, b);
  wire x = 'x;

  // $changed returns a property type (!ltl.property) because it uses ltl.and/ltl.or/ltl.not.
  // Sequence concatenation (##1) requires sequence types (i1 or !ltl.sequence).
  // CHECK: error: property type cannot be used in sequence concatenation; consider restructuring the assertion to use the property as a consequent
  assume property (@(posedge clk) b !== x ##1 $changed(b));

endmodule

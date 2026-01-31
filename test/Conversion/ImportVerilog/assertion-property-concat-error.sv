// RUN: circt-verilog --ir-moore --no-uvm-auto-include %s 2>&1 | FileCheck %s
// REQUIRES: slang

// Test that $changed is lowered to a sequence-compatible value in concatenation.

module PropertyConcatError(input logic clk, a, b);
  wire x = 'x;

  // $changed should lower to a sampled comparison that can be concatenated.
  // CHECK: moore.past
  // CHECK: moore.eq
  // CHECK: ltl.concat
  assume property (@(posedge clk) b !== x ##1 $changed(b));

endmodule

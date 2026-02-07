// RUN: circt-verilog --ir-moore --no-uvm-auto-include %s 2>&1 | FileCheck %s
// REQUIRES: slang

// Test that $changed is lowered to a sequence-compatible value in concatenation.

module PropertyConcatError(input logic clk, a, b);
  wire x = 'x;

  // $changed now uses register-based tracking via procedure.
  // CHECK: moore.procedure always
  // CHECK: moore.eq
  // CHECK: moore.not
  // CHECK: moore.blocking_assign
  // CHECK: ltl.concat
  assume property (@(posedge clk) b !== x ##1 $changed(b));

endmodule

// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

module SvaBoundedUnaryPropertyError(input logic clk, a, b);
  logic c;

  // Enabled $past without explicit clocking should lower via implicit sampled
  // state instead of producing a frontend error.
  assert property ($past(a, 1, b) |-> c);
endmodule

// CHECK-LABEL: moore.module @SvaBoundedUnaryPropertyError
// CHECK: moore.variable
// CHECK: moore.procedure always
// CHECK: moore.conditional
// CHECK: verif.assert

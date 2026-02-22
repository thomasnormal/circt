// RUN: circt-translate --import-verilog --verify-diagnostics %s
// REQUIRES: slang

module SvaBoundedUnaryPropertyError(input logic clk, a, b);
  property p;
    @(posedge clk) a |-> b;
  endproperty

  // Property-valued always forms still require dedicated lowering.
  // expected-error @below {{always on property expressions is not yet supported}}
  assert property (always p);
endmodule

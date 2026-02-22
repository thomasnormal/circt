// RUN: circt-translate --import-verilog --verify-diagnostics %s
// REQUIRES: slang

module SvaBoundedUnaryPropertyError(input logic clk, a, b);
  property p;
    @(posedge clk) a |-> b;
  endproperty

  // Property-valued nexttime forms still require dedicated lowering.
  // expected-error @below {{nexttime on property expressions is not yet supported}}
  assert property (nexttime p);
endmodule

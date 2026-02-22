// RUN: circt-translate --import-verilog --verify-diagnostics %s
// REQUIRES: slang

module SvaBoundedUnaryPropertyError(input logic clk, a, b);
  property p;
    @(posedge clk) a |-> b;
  endproperty

  // `$past` enable expressions without explicit clocking are unsupported.
  // expected-error @below {{unsupported $past enable expression without explicit clocking}}
  assert property ($past(a, 1, b));
endmodule

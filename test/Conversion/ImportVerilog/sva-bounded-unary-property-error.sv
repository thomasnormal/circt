// RUN: circt-translate --import-verilog --verify-diagnostics %s
// REQUIRES: slang

module SvaBoundedUnaryPropertyError(input logic clk, a, b);
  property p;
    @(posedge clk) a |-> b;
  endproperty

  // Previously this could reach MLIR verifier failure due to illegal
  // `ltl.delay` on `!ltl.property`. Emit a frontend diagnostic instead.
  // expected-error @below {{bounded eventually on property expressions is not yet supported}}
  assert property (eventually [1:2] p);
endmodule

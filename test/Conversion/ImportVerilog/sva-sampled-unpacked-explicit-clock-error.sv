// RUN: circt-translate --import-verilog --verify-diagnostics %s
// REQUIRES: slang

module SvaSampledUnpackedExplicitClockError(input logic clk);
  logic [1:0] s [2];

  // Unsupported sampled operand types should diagnose cleanly, not crash.
  // expected-error @below {{expression of type '!moore.uarray<2 x l2>' cannot be cast to a simple bit vector}}
  assert property ($changed(s, @(posedge clk)));
endmodule

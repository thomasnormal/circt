// RUN: circt-translate --import-verilog --verify-diagnostics %s
// REQUIRES: slang

module SvaSampledUnpackedExplicitClockError(input logic clk);
  logic [1:0] s [2];

  // $rose/$fell still require scalar sampled operands.
  // expected-error @below {{unsupported sampled value type for $rose}}
  assert property ($rose(s, @(posedge clk)));
endmodule

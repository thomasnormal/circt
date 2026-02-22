// RUN: circt-translate --import-verilog --verify-diagnostics %s
// REQUIRES: slang

module SvaSampledUnpackedExplicitClockError(input logic clk);
  logic [1:0] s [2];

  // Unsupported sampled operand types should diagnose cleanly, not crash.
  // expected-error @below {{unsupported sampled value type for $changed}}
  assert property ($changed(s, @(posedge clk)));
endmodule

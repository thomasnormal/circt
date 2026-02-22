// RUN: not circt-translate --import-verilog %s 2>&1 | FileCheck %s
// REQUIRES: slang

module SvaSampledUnpackedExplicitClockError(input logic clk);
  int s[];

  // Dynamic arrays still cannot be converted to a sampled scalar for
  // $rose/$fell helper lowering.
  // CHECK: error: expression of type '!moore.open_uarray<i32>' cannot be cast to a simple bit vector
  assert property ($rose(s, @(posedge clk)));
endmodule

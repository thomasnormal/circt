// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

module SvaSampledUnpackedExplicitClockError(input logic clk);
  int aa[int];

  // Associative arrays should lower via sampled-value helper state.
  // CHECK-LABEL: moore.module @SvaSampledUnpackedExplicitClockError
  // CHECK: moore.procedure always
  // CHECK: verif.assert
  assert property ($rose(aa, @(posedge clk)));
endmodule

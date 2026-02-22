// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSampledUnpackedExplicitClock(input logic clk_a, input logic clk_b);
  logic [1:0] s [2];

  // Force helper-procedure lowering by using a different explicit sampled clock.
  // CHECK: moore.procedure always
  // CHECK: moore.uarray_cmp eq
  // CHECK: moore.not
  // CHECK: verif.assert
  assert property (@(posedge clk_a) $changed(s, @(posedge clk_b)));

  // CHECK: moore.uarray_cmp eq
  // CHECK: verif.assert
  assert property (@(posedge clk_a) $stable(s, @(posedge clk_b)));
endmodule

// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module sva_strong_weak(input logic clk, a, b);
  // CHECK-LABEL: moore.module @sva_strong_weak

  // CHECK: ltl.concat
  // CHECK: verif.assert
  assert property (@(posedge clk) strong(a ##1 b));

  // CHECK: ltl.concat
  // CHECK: verif.assert
  assert property (@(posedge clk) weak(a ##1 b));
endmodule

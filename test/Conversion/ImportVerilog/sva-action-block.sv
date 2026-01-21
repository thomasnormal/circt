// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Ensure concurrent assertions with action blocks still import the property.
module ActionBlockAssert(input logic clk, rst, a, b);
  // CHECK-LABEL: moore.module @ActionBlockAssert
  // CHECK: verif.assert
  assert property (@(posedge clk) disable iff (rst) a |=> b) else $error("fail");
endmodule

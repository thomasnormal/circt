// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSampledVectorRoseFellLsb(input logic clk, input logic [1:0] v);
  // For packed vectors, $rose/$fell sample bit 0 transitions.
  // CHECK-LABEL: moore.module @SvaSampledVectorRoseFellLsb
  // CHECK: moore.extract
  // CHECK: moore.past
  // CHECK: moore.case_eq
  // CHECK-NOT: moore.bool_cast %{{.*}} : l2 -> l1
  assert property (@(posedge clk) $rose(v));
  assert property (@(posedge clk) $fell(v));
endmodule

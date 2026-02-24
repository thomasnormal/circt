// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSampledVectorRoseFellExplicitClock(input logic clk,
                                             input logic [1:0] v);
  logic rose_v, fell_v;

  // Explicit-clock sampled-value helpers should also use bit 0 for vectors.
  // CHECK-LABEL: moore.module @SvaSampledVectorRoseFellExplicitClock
  // CHECK: moore.wait_event
  // CHECK: moore.extract
  // CHECK: moore.case_eq
  // CHECK-NOT: moore.bool_cast %{{.*}} : l2 -> l1
  assign rose_v = $rose(v, @(posedge clk));
  assign fell_v = $fell(v, @(posedge clk));
endmodule

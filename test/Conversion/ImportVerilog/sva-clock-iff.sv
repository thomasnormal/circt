// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Test clocking events with iff conditions.

// CHECK-LABEL: moore.module @sva_clock_iff
module sva_clock_iff(
  input logic clk,
  input logic gate,
  input logic a
);
  // CHECK: [[GATED:%.*]] = ltl.and {{%.*}}, {{%.*}} : i1, i1
  // CHECK: ltl.clock [[GATED]],{{ *}}posedge {{%.*}} : i1
  assert property (@(posedge clk iff gate) a);
endmodule

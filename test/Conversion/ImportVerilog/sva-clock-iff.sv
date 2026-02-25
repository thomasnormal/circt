// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Test clocking events with iff conditions.

// CHECK-LABEL: moore.module @sva_clock_iff
module sva_clock_iff(
  input logic clk,
  input logic gate,
  input logic a
);
  // CHECK: [[A:%.*]] = moore.to_builtin_bool {{%.*}} : l1
  // CHECK: [[GATE_INT:%.*]] = moore.logic_to_int {{%.*}} : l1
  // CHECK: [[GATE:%.*]] = moore.to_builtin_bool [[GATE_INT]] : i1
  // CHECK: [[CLK:%.*]] = moore.to_builtin_bool {{%.*}} : l1
  // CHECK: [[GATED:%.*]] = comb.and bin [[GATE]], [[A]] : i1
  // CHECK: verif.clocked_assert [[GATED]], posedge [[CLK]] : i1
  assert property (@(posedge clk iff gate) a);
endmodule

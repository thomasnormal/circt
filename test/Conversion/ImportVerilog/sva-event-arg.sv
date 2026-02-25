// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Test event-typed sequence arguments.

// CHECK-LABEL: moore.module @sva_event_arg
module sva_event_arg(
  input logic clk,
  input logic a,
  input logic b
);
  sequence seq_event(event e, logic x);
    @e x;
  endsequence

  // CHECK: [[A:%.*]] = moore.to_builtin_bool {{%.*}} : l1
  // CHECK: [[CLK:%.*]] = moore.to_builtin_bool {{%.*}} : l1
  // CHECK: verif.clocked_assert [[A]], posedge [[CLK]] : i1
  assert property (seq_event(posedge clk, a));

  // CHECK: [[B:%.*]] = moore.to_builtin_bool {{%.*}} : l1
  // CHECK: verif.clocked_assert [[B]], negedge [[CLK]] : i1
  assert property (seq_event(negedge clk, b));
endmodule

// RUN: circt-translate --import-verilog %s | FileCheck %s
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
  // CHECK: [[CLOCKED_A:%.*]] = ltl.clock [[A]],{{ *}}posedge [[CLK]]{{.*}} : i1
  // CHECK: verif.assert [[CLOCKED_A]] : !ltl.sequence
  assert property (seq_event(posedge clk, a));

  // CHECK: [[B:%.*]] = moore.to_builtin_bool {{%.*}} : l1
  // CHECK: [[CLK2:%.*]] = moore.to_builtin_bool {{%.*}} : l1
  // CHECK: [[CLOCKED_B:%.*]] = ltl.clock [[B]],{{ *}}negedge [[CLK2]]{{.*}} : i1
  // CHECK: verif.assert [[CLOCKED_B]] : !ltl.sequence
  assert property (seq_event(negedge clk, b));
endmodule

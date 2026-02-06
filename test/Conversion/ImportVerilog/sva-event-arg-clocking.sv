// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Test event-typed assertion port used as explicit clocking argument.

// CHECK-LABEL: moore.module @sva_event_arg_clocking
module sva_event_arg_clocking(
  input logic clk,
  input logic a
);
  sequence seq_past(event e, logic x);
    $past(x, 1, @(e));
  endsequence

  // CHECK: moore.procedure
  // CHECK: moore.wait_event
  // CHECK: verif.assert
  assert property (seq_past(posedge clk, a));
endmodule

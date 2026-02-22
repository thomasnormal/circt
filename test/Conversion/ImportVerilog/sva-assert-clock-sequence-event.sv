// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaAssertClockSequenceEvent(input logic clk, a, b, c);
  default clocking cb @(posedge clk); endclocking

  sequence s;
    a ##1 b;
  endsequence

  // Sequence-valued assertion clocking events should lower via sequence
  // match detection instead of requiring a plain 1-bit signal expression.
  // CHECK-LABEL: moore.module @SvaAssertClockSequenceEvent
  // CHECK: ltl.matched
  // CHECK: ltl.clock
  // CHECK: verif.assert
  assert property (@s c);
endmodule


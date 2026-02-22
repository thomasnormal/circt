// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaAssertClockNamedEvent(input logic clk, req, ack, c, d);
  default clocking cb @(posedge clk); endclocking

  event e;
  sequence s;
    req ##1 ack;
  endsequence

  // Named event clocking should be supported directly in assertion timing.
  // CHECK-LABEL: moore.module @SvaAssertClockNamedEvent
  // CHECK: moore.event_triggered
  // CHECK: verif.assert
  assert property (@(e) c);

  // Mixed sequence + named-event clocking should also lower.
  // CHECK: ltl.matched
  // CHECK: moore.event_triggered
  // CHECK: ltl.or
  // CHECK: verif.assert
  assert property (@(s or e) d);
endmodule

// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSequenceEventListGlobalClock(input logic clk, req, ack);
  global clocking gclk @(posedge clk); endclocking

  sequence s;
    req ##1 ack;
  endsequence

  // Mixed sequence + $global_clock event lists should resolve through the
  // scope global clocking declaration.
  // CHECK-LABEL: moore.module @SvaSequenceEventListGlobalClock
  // CHECK: moore.wait_event
  // CHECK-DAG: moore.detect_event posedge
  always @(s or $global_clock) begin
  end
endmodule

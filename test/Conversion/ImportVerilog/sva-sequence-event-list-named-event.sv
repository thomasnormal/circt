// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSequenceEventListNamedEvent(input logic clk, req, ack);
  default clocking cb @(posedge clk); endclocking

  event e;
  sequence s;
    req ##1 ack;
  endsequence

  // Mixed sequence + named-event lists should lower without requiring a
  // 1-bit conversion of the named event expression.
  // CHECK-LABEL: moore.module @SvaSequenceEventListNamedEvent
  // CHECK: moore.wait_event
  // CHECK-DAG: moore.detect_event any
  // CHECK-DAG: moore.detect_event posedge
  always @(s or e) begin
  end
endmodule

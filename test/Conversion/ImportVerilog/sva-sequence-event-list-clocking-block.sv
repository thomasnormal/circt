// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSequenceEventListClockingBlock(input logic clk, req, ack);
  clocking cb @(posedge clk); endclocking

  sequence s;
    req ##1 ack;
  endsequence

  // Mixed sequence + clocking-block event lists should lower by resolving the
  // clocking block event rather than treating the clocking block symbol as an
  // arbitrary expression.
  // CHECK-LABEL: moore.module @SvaSequenceEventListClockingBlock
  // CHECK: moore.wait_event
  // CHECK-DAG: moore.detect_event posedge
  always @(s or cb) begin
  end
endmodule

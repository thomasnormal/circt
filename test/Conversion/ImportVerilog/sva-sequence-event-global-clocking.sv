// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSequenceEventGlobalClocking(input logic clk, req, ack, c);
  global clocking gclk @(posedge clk); endclocking

  sequence s;
    req ##1 ack;
  endsequence

  // Unclocked sequence event controls should fall back to global clocking when
  // default clocking is not declared.
  // CHECK-LABEL: moore.module @SvaSequenceEventGlobalClocking
  // CHECK: moore.wait_event
  // CHECK: moore.detect_event posedge
  always @(s) begin
  end

  // Sequence-valued assertion clocking events should also use global clocking.
  // CHECK: ltl.matched
  // CHECK: verif.assert
  assert property (@s c);
endmodule


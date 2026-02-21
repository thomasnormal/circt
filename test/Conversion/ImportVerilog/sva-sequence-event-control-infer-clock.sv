// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module m(input logic clk, req, ack);
  sequence s;
    req ##1 ack;
  endsequence

  // Unclocked sequence event controls in a mixed event list should infer their
  // clock from a uniform signal event clock.
// CHECK-LABEL: moore.module @m
// CHECK: moore.wait_event
// CHECK: moore.detect_event posedge
// CHECK: moore.event_sources =
  always @(s or posedge clk) begin
  end
endmodule

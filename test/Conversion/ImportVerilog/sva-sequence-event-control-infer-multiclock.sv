// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module m(input logic clk, rst, req, ack);
  sequence s;
    req ##1 ack;
  endsequence

  // Unclocked sequence events in mixed event lists should infer sampling from
  // the listed signal event clocks, even when they are non-uniform.
// CHECK-LABEL: moore.module @m
// CHECK: moore.wait_event
// CHECK-COUNT-2: moore.detect_event any
// CHECK: moore.event_sources =
always @(s or posedge clk or negedge rst) begin
end
endmodule

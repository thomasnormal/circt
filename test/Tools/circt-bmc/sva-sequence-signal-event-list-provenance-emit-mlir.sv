// RUN: circt-verilog --no-uvm-auto-include --ir-llhd --timescale=1ns/1ns --single-unit %s | FileCheck %s
// REQUIRES: slang

module sva_sequence_signal_event_list_provenance_emit_mlir(
    input logic clk_a, clk_b, a, b);
  bit seen = 0;

  sequence s;
    @(posedge clk_a) a;
  endsequence

  always @(s or posedge clk_b iff b)
    seen <= 1'b1;

  // Keep behavior non-trivial to ensure event control is retained.
  assert property (@(posedge clk_b) b || !seen);
endmodule

// CHECK: hw.module @sva_sequence_signal_event_list_provenance_emit_mlir
// CHECK-SAME: moore.mixed_event_sources
// CHECK-SAME: "sequence"
// CHECK-SAME: "signal[0]:posedge:iff"

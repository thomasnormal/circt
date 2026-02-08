// RUN: circt-verilog --no-uvm-auto-include --ir-llhd --timescale=1ns/1ns --single-unit %s | FileCheck %s
// REQUIRES: slang

module sva_sequence_event_list_provenance_emit_mlir(
    input logic clk_a, clk_b, a, b);
  bit seen = 0;

  sequence s1;
    @(posedge clk_a) a;
  endsequence

  sequence s2;
    @(posedge clk_b) b;
  endsequence

  always @(s1 or s2)
    seen <= 1'b1;

  // Keep behavior non-trivial so event control is retained.
  assert property (@(posedge clk_a) a || !seen);
endmodule

// CHECK: hw.module @sva_sequence_event_list_provenance_emit_mlir
// CHECK-SAME: moore.event_sources
// CHECK-SAME: "sequence[0]"
// CHECK-SAME: "sequence[1]"

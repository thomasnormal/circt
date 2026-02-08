// RUN: circt-verilog --no-uvm-auto-include --ir-llhd --timescale=1ns/1ns --single-unit %s | \
// RUN:   circt-bmc -b 20 --ignore-asserts-until=3 --module=sva_sequence_event_iff_unsat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_sequence_event_iff_unsat(input logic clk);
  bit a = 0;
  bit en = 0;
  int via_seq_iff = 0;
  int via_ref = 0;

  always @(posedge clk) begin
    // Reference model uses sampled values at this edge.
    via_ref <= via_ref + (a && en);
    a <= ~a;
    en <= ~en;
  end

  sequence s;
    @(posedge clk) a;
  endsequence

  always @(s iff en)
    via_seq_iff <= via_seq_iff + 1;

  assert property (@(posedge clk) via_seq_iff == via_ref);
endmodule

// CHECK: BMC_RESULT=UNSAT

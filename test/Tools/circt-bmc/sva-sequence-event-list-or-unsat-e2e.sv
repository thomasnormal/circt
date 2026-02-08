// RUN: circt-verilog --no-uvm-auto-include --ir-llhd --timescale=1ns/1ns --single-unit %s | \
// RUN:   circt-bmc -b 12 --ignore-asserts-until=2 --module=sva_sequence_event_list_or_unsat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_sequence_event_list_or_unsat(input logic clk);
  bit hit = 0;

  sequence s1;
    @(posedge clk) 1'b1;
  endsequence

  sequence s2;
    @(posedge clk) 1'b0;
  endsequence

  always @(s1 or s2)
    hit <= ~hit;

  // s1 matches every posedge, so hit must change on the next sampled cycle.
  assert property (@(posedge clk) 1'b1 |=> $changed(hit));
endmodule

// CHECK: BMC_RESULT=UNSAT

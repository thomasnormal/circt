// RUN: circt-verilog --no-uvm-auto-include --ir-llhd --timescale=1ns/1ns --single-unit %s | \
// RUN:   circt-bmc -b 16 --ignore-asserts-until=2 --module=sva_sequence_signal_event_list_noedge_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_sequence_signal_event_list_noedge_sat(input logic clk);
  bit b = 0;
  bit hit = 0;

  // Never matches; wakeups should come from the no-edge signal event `b`.
  sequence s_never;
    @(posedge clk) 1'b0;
  endsequence

  always @(posedge clk)
    b <= ~b;

  // Regression: mixed lists with no-edge signal event should lower.
  always @(s_never or b)
    hit <= ~hit;

  // Must become reachable once b toggles and triggers the event list.
  cover property (@(posedge clk) $changed(hit));
endmodule

// CHECK: BMC_RESULT=SAT

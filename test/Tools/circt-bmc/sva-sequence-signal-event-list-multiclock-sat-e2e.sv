// RUN: circt-verilog --no-uvm-auto-include --ir-llhd --timescale=1ns/1ns --single-unit %s | \
// RUN:   circt-bmc -b 12 --allow-multi-clock --assume-known-inputs --module=sva_sequence_signal_event_list_multiclock_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_sequence_signal_event_list_multiclock_sat(
    input logic clk_a, clk_b, a, b);
  bit seen = 0;

  sequence s;
    @(posedge clk_a) a;
  endsequence

  // Mixed sequence/signal event-list on different clocks.
  always @(s or posedge clk_b iff b)
    seen <= 1'b1;

  // Suppress sequence-triggered wakeups and force signal-event wakeups.
  assume property (@(posedge clk_a) !a);
  assume property (@(posedge clk_b) b);

  // If the mixed multiclock signal-event path is active, this is violated.
  assert property (@(posedge clk_b) !seen);
endmodule

// CHECK: BMC_RESULT=SAT

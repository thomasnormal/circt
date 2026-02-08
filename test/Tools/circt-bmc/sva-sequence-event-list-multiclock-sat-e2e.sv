// RUN: circt-verilog --no-uvm-auto-include --ir-llhd --timescale=1ns/1ns --single-unit %s | \
// RUN:   circt-bmc -b 12 --allow-multi-clock --assume-known-inputs --module=sva_sequence_event_list_multiclock_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_sequence_event_list_multiclock_sat(
    input logic clk_a, clk_b, a, b);
  bit seen = 0;

  sequence s1;
    @(posedge clk_a) a;
  endsequence

  sequence s2;
    @(posedge clk_b) b;
  endsequence

  // Different-clock sequence event-list support under procedural timing control.
  always @(s1 or s2)
    seen <= 1'b1;

  // Force recurring sequence matches on clk_a and require seen to remain false.
  // This should be violated once the event-list wakeup happens.
  assume property (@(posedge clk_a) a);
  assert property (@(posedge clk_a) !seen);
endmodule

// CHECK: BMC_RESULT=SAT

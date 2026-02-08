// RUN: circt-verilog --no-uvm-auto-include --ir-llhd --timescale=1ns/1ns --single-unit %s | \
// RUN:   circt-bmc -b 20 --ignore-asserts-until=3 --module=sva_sequence_signal_event_list_equivalent_clock_unsat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_sequence_signal_event_list_equivalent_clock_unsat(input logic clk);
  bit a = 0;
  bit b = 0;
  int via_mixed = 0;
  int via_ref = 0;

  always @(posedge clk) begin
    // Reference model uses sampled values of a/b at this edge.
    via_ref <= via_ref + (a || b);
    a <= ~a;
    b <= ~b;
  end

  sequence s;
    @(posedge clk) a;
  endsequence

  // Mixed event list with equivalent clock edge and signal iff should trigger
  // whenever either sampled sequence match or sampled iff holds.
  always @(s or posedge clk iff b)
    via_mixed <= via_mixed + 1;

  assert property (@(posedge clk) via_mixed == via_ref);
endmodule

// CHECK: BMC_RESULT=UNSAT

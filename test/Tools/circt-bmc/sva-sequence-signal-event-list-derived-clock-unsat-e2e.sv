// RUN: circt-verilog --no-uvm-auto-include --ir-llhd --timescale=1ns/1ns --single-unit %s | \
// RUN:   circt-bmc -b 24 --allow-multi-clock --ignore-asserts-until=3 --module=sva_sequence_signal_event_list_derived_clock_unsat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_sequence_signal_event_list_derived_clock_unsat(input logic clk);
  bit dclk = 0;
  bit a = 0;
  bit b = 0;
  int via_mixed = 0;
  int via_ref = 0;

  // Derived clock used by both the sequence and explicit signal event arm.
  always @(posedge clk) begin
    dclk <= ~dclk;
    a <= ~a;
    b <= ~b;
  end

  sequence s;
    @(posedge dclk) a;
  endsequence

  // Mixed sequence/signal event-list on a derived clock.
  always @(s or posedge dclk iff b)
    via_mixed <= via_mixed + 1;

  // Reference sampled-value model on the same derived clock.
  always @(posedge dclk)
    via_ref <= via_ref + (a || b);

  assert property (@(posedge dclk) via_mixed == via_ref);
endmodule

// CHECK: BMC_RESULT=UNSAT

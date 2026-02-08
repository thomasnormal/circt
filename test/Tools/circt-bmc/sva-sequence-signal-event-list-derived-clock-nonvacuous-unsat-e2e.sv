// RUN: circt-verilog --no-uvm-auto-include --ir-llhd --timescale=1ns/1ns --single-unit %s | \
// RUN:   circt-bmc -b 24 --allow-multi-clock --ignore-asserts-until=3 --module=sva_sequence_signal_event_list_derived_clock_nonvacuous_unsat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_sequence_signal_event_list_derived_clock_nonvacuous_unsat(
    input logic clk);
  bit dclk = 0;
  bit a = 0;
  bit b = 0;
  int via_mixed = 0;
  int via_ref = 0;

  always @(posedge clk) begin
    dclk <= ~dclk;
    a <= ~a;
    b <= ~b;
  end

  sequence s;
    @(posedge dclk) a;
  endsequence

  always @(s or posedge dclk iff b)
    via_mixed <= via_mixed + 1;

  always @(posedge dclk)
    via_ref <= via_ref + (a || b);

  // Non-vacuous forcing: require at least one reference wakeup by final step.
  assume final (via_ref > 0);
  assert property (@(posedge dclk) via_mixed == via_ref);
endmodule

// CHECK: BMC_RESULT=UNSAT

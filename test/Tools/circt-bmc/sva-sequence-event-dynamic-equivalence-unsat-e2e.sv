// RUN: circt-verilog --no-uvm-auto-include --ir-llhd --timescale=1ns/1ns --single-unit %s | \
// RUN:   circt-bmc -b 20 --ignore-asserts-until=3 --module=sva_sequence_event_dynamic_equivalence_unsat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_sequence_event_dynamic_equivalence_unsat(input logic clk);
  bit a = 0;
  int via_seq = 0;
  int via_if = 0;

  always @(posedge clk) begin
    a <= ~a;
    if (a)
      via_if <= via_if + 1;
  end

  sequence s;
    @(posedge clk) a;
  endsequence

  always @(s)
    via_seq <= via_seq + 1;

  // always @(s) should match explicit sampled if(a) behavior.
  assert property (@(posedge clk) via_seq == via_if);
endmodule

// CHECK: BMC_RESULT=UNSAT

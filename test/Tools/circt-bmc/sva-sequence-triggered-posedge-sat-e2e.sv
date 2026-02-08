// RUN: circt-verilog --no-uvm-auto-include --ir-llhd --timescale=1ns/1ns --single-unit %s | \
// RUN:   circt-bmc -b 12 --ignore-asserts-until=4 --module=sva_sequence_triggered_posedge_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_sequence_triggered_posedge_sat(input logic clk);
  bit a = 0;
  bit seen = 0;

  sequence s;
    @(posedge clk) a;
  endsequence

  // Force a to become and stay 1 so s.triggered can rise after startup.
  always @(posedge clk)
    a <= 1'b1;

  // Lowered through ltl.triggered for sequence-valued .triggered.
  always @(posedge s.triggered)
    seen <= 1'b1;

  // This intentionally fails for some traces, locking in end-to-end support
  // for sequence .triggered in procedural timing controls.
  assert property (@(posedge clk) seen);
endmodule

// CHECK: BMC_RESULT=SAT

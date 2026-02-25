// RUN: circt-verilog --no-uvm-auto-include --ir-llhd --timescale=1ns/1ns --single-unit %s | \
// RUN:   circt-bmc --run-smtlib -b 3 --ignore-asserts-until=0 --module=sva_ended_e2e - | FileCheck %s
// REQUIRES: slang
// REQUIRES: z3

module sva_ended_e2e(input logic clk, input logic a, input logic b);
  sequence s;
    @(posedge clk) a ##1 b;
  endsequence

  // There is always a counterexample where the sequence does not end.
  // This test locks in end-to-end support for sequence `.ended`.
  assert property (@(posedge clk) s.ended);
endmodule

// CHECK: BMC_RESULT=SAT

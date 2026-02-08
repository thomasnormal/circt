// RUN: circt-verilog --no-uvm-auto-include --ir-llhd --timescale=1ns/1ns --single-unit %s | \
// RUN:   circt-bmc -b 10 --ignore-asserts-until=2 --module=sva_past_nonoverlap_unsat - | FileCheck %s --check-prefix=IMPLICIT
// RUN: circt-verilog --no-uvm-auto-include --ir-llhd --timescale=1ns/1ns --single-unit %s | \
// RUN:   circt-bmc -b 10 --ignore-asserts-until=2 --module=sva_past_explicit_same_clock_nonoverlap_unsat - | FileCheck %s --check-prefix=EXPLICIT
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_past_nonoverlap_unsat(input logic clk);
  bit val = 0;
  always @(posedge clk)
    val <= ~val;

  assert property (@(posedge clk) 1'b1 |=> ($past(val) != val));
endmodule

module sva_past_explicit_same_clock_nonoverlap_unsat(input logic clk);
  bit val = 0;
  always @(posedge clk)
    val <= ~val;

  assert property (@(posedge clk) 1'b1 |=> ($past(val, 1, @(posedge clk)) != val));
endmodule

// IMPLICIT: BMC_RESULT=UNSAT
// EXPLICIT: BMC_RESULT=UNSAT

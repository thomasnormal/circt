// RUN: circt-verilog --no-uvm-auto-include --ir-llhd --timescale=1ns/1ns --single-unit %s | \
// RUN:   circt-bmc --run-smtlib -b 10 --ignore-asserts-until=1 --module top --assume-known-inputs - | \
// RUN:   FileCheck %s --check-prefix=PASS
// RUN: circt-verilog --no-uvm-auto-include --ir-llhd --timescale=1ns/1ns --single-unit -DFAIL %s | \
// RUN:   circt-bmc --run-smtlib -b 10 --ignore-asserts-until=1 --module top --assume-known-inputs - | \
// RUN:   FileCheck %s --check-prefix=FAIL
// REQUIRES: slang
// REQUIRES: z3

// Parity lock for yosys/tests/sva/basic00.sv:
// - PASS profile (|=>) must prove UNSAT.
// - FAIL profile (|->) must produce SAT.
module top(input clk, reset, antecedent, output reg consequent);
  always @(posedge clk)
    consequent <= reset ? 1'b0 : antecedent;

`ifdef FAIL
  assert property (@(posedge clk) disable iff (reset) antecedent |-> consequent);
`else
  assert property (@(posedge clk) disable iff (reset) antecedent |=> consequent);
`endif
endmodule

// PASS: BMC_RESULT=UNSAT
// FAIL: BMC_RESULT=SAT

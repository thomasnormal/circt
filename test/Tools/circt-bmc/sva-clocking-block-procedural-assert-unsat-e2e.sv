// RUN: circt-verilog --no-uvm-auto-include --ir-llhd --timescale=1ns/1ns --single-unit %s | \
// RUN:   circt-bmc -b 10 --ignore-asserts-until=2 --module=sva_clocking_block_procedural_assert_unsat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_clocking_block_procedural_assert_unsat(input logic clk);
  bit val = 0;
  clocking cb @(posedge clk); endclocking

  always @(posedge clk)
    val <= ~val;

  // This exercises assertion hoisting from a timed statement whose event
  // control is a clocking block reference.
  always @(cb)
    assert property ($past(val) != val);
endmodule

// CHECK: BMC_RESULT=UNSAT

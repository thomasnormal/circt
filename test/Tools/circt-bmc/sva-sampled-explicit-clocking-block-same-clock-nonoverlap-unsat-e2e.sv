// RUN: circt-verilog --no-uvm-auto-include --ir-llhd --timescale=1ns/1ns --single-unit %s | \
// RUN:   circt-bmc -b 10 --ignore-asserts-until=2 --module=sva_rose_explicit_clocking_block_same_clock_nonoverlap_unsat - | FileCheck %s --check-prefix=ROSE
// RUN: circt-verilog --no-uvm-auto-include --ir-llhd --timescale=1ns/1ns --single-unit %s | \
// RUN:   circt-bmc -b 10 --ignore-asserts-until=2 --module=sva_past_explicit_clocking_block_same_clock_nonoverlap_unsat - | FileCheck %s --check-prefix=PAST
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_rose_explicit_clocking_block_same_clock_nonoverlap_unsat(
    input logic clk);
  int cyc = 0;
  bit val = 0;
  clocking cb @(posedge clk); endclocking

  always @(posedge clk) begin
    cyc <= cyc + 1;
    val <= ~val;
  end

  assert property (@(posedge clk) cyc % 2 == 0 |=> $rose(val, @(cb)));
endmodule

module sva_past_explicit_clocking_block_same_clock_nonoverlap_unsat(
    input logic clk);
  bit val = 0;
  clocking cb @(posedge clk); endclocking

  always @(posedge clk)
    val <= ~val;

  assert property (@(posedge clk) 1'b1 |=> ($past(val, 1, @(cb)) != val));
endmodule

// ROSE: BMC_RESULT=UNSAT
// PAST: BMC_RESULT=UNSAT

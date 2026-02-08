// RUN: circt-verilog --ir-llhd --timescale=1ns/1ns --single-unit %s | \
// RUN:   circt-bmc -b 10 --ignore-asserts-until=1 --module=top - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module top(input logic clk);
  int cyc = 0;
  bit val = 0;

  always @(posedge clk) begin
    cyc <= cyc + 1;
    val = ~val;
  end

  assert property (@(posedge clk) cyc % 2 == 0 |=> $rose(val));
endmodule

// CHECK: BMC_RESULT=UNSAT

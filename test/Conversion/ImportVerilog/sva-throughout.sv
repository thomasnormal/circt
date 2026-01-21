// RUN: circt-verilog --ir-hw %s | FileCheck %s

module top(
  input logic clk,
  input logic a,
  input logic b,
  input logic c,
  input logic d
);
  default clocking @(posedge clk); endclocking

  assert property (
    a |=> b throughout (c ##1 d)
  );

  // CHECK: ltl.repeat {{.*}}, 2, 0 : i1
  // CHECK: ltl.intersect
endmodule

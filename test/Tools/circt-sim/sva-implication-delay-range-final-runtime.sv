// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=80000000 2>&1 | FileCheck %s
// CHECK: Simulation completed
// CHECK-NOT: SVA assertion failed at time

// Runtime end-of-simulation semantics: for weak bounded implication, when
// `a |-> ##[1:2] b` is triggered but simulation ends before any allowed
// consequent sample can be observed, the open obligation is not forced to fail
// at finalization.

module top;
  reg clk;
  reg a, b;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b0;
    b = 1'b0;

    @(posedge clk); // cycle 1
    a = 1'b1;

    @(posedge clk); // cycle 2: antecedent sampled high
    #1;
    $finish;        // end before next sampled cycle (no chance to satisfy b)
  end

  assert property (@(posedge clk) a |-> ##[1:2] b);
endmodule

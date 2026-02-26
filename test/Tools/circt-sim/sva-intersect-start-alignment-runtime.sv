// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Runtime semantics: `intersect` requires a common start and end time.
// `##[1:2] a` and `##1 b` can both end in the same cycle while referring to
// different starts; that must not count as an intersect match.

module top;
  reg clk;
  reg a, b;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b1;
    b = 1'b1;

    @(posedge clk); // cycle 1
    a = 1'b0;
    b = 1'b1;

    @(posedge clk); // cycle 2
    a = 1'b0;
    b = 1'b0;

    @(posedge clk); // cycle 3
    #1 $finish;
  end

  assert property (@(posedge clk) ((##[1:2] a) intersect (##1 b)));
endmodule

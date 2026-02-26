// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=80000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Runtime semantics: `a[->2]` should match when the second observed `a=1`
// occurs (final hit at current sample). So `not (a[->2])` must fail.

module top;
  reg clk;
  reg a;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b1;
    @(posedge clk);
    a = 1'b0;
    @(posedge clk);
    a = 1'b1;
    @(posedge clk);
    @(posedge clk);
    $finish;
  end

  assert property (@(posedge clk) not (a [-> 2]));
endmodule

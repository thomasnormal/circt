// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=80000000 2>&1 | FileCheck %s
// CHECK-NOT: SVA assertion failed
// CHECK: Simulation completed

// Runtime semantics: `a until_with b` should pass when `a` stays high until
// the overlap cycle where `b` becomes high.

module top;
  reg clk;
  reg a, b;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b1;
    b = 1'b0;
    @(posedge clk);
    @(posedge clk);
    b = 1'b1;
    @(posedge clk);
    b = 1'b0;
    @(posedge clk);
    $finish;
  end

  assert property (@(posedge clk) a until_with b);
endmodule

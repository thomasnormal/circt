// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Regression: always [m:$] over a property must fail when the operand property
// is violated after the lower bound.

module top;
  reg clk;
  reg b;

  property p;
    @(posedge clk) b;
  endproperty

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    b = 1'b1;
    @(posedge clk);
    @(posedge clk);
    b = 1'b0;
    @(posedge clk);
    $finish;
  end

  assert property (@(posedge clk) always [1:$] p);
endmodule

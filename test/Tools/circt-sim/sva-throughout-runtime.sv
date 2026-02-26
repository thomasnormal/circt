// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=60000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Runtime semantics: `a throughout b` requires `a` to hold for the duration of
// `b`. With `b=1` and `a=0` at sampled edges, this should fail.

module top;
  reg clk;
  reg a, b;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b0;
    b = 1'b1;
    @(posedge clk);
    @(posedge clk);
    @(posedge clk);
    $finish;
  end

  assert property (@(posedge clk) a throughout b);
endmodule

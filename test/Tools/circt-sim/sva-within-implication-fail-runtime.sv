// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

module top;
  reg clk;
  reg t, a, b;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    t = 1'b1;
    a = 1'b1;
    b = 1'b0;
    repeat (4) @(posedge clk);
    $finish;
  end

  assert property (@(posedge clk) t |-> (a within (##1 b)));
endmodule

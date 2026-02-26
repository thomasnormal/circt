// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=70000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Runtime semantics: strong(b[->2]) must fail when b is true only once.

module top;
  reg clk;
  reg b;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    b = 1'b0;
    @(posedge clk);
    b = 1'b1;
    @(posedge clk);
    b = 1'b0;
    @(posedge clk);
    $finish;
  end

  assert property (@(posedge clk) strong(b[->2]));
endmodule

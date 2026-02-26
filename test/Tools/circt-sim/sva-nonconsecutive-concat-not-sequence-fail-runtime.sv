// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=120000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Runtime semantics: nonconsecutive-repeat endpoints must align with ##1
// concatenation in direct sequence properties.

module top;
  reg clk;
  reg b;
  reg c;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    b = 0;
    c = 0;
    @(negedge clk); b = 1; c = 0;
    @(negedge clk); b = 0; c = 1;
    @(negedge clk); c = 0;
    @(negedge clk); $finish;
  end

  assert property (@(posedge clk) not (b[=1] ##1 c));
endmodule

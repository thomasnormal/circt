// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=60000000 2>&1 | FileCheck %s
// CHECK: SVA assumption failed at time
// CHECK: SVA assumption failure(s)
// CHECK: exit code 1

// Runtime semantics: clocked assume should be checked like concurrent SVA.
// This test intentionally violates the assumption on a sampled edge.

module top;
  reg clk;
  reg a;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b1;
    @(posedge clk); // pass
    a = 1'b0;
    @(posedge clk); // fail
    $finish;
  end

  assume property (@(posedge clk) a);
endmodule

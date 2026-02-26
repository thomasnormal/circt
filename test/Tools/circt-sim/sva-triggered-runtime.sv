// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=60000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Runtime semantics: `s.triggered` should report that sequence `s` started on
// the previous sampled cycle, even if the full sequence has not matched yet.

module top;
  reg clk;
  reg a, b;

  sequence s;
    a ##1 b;
  endsequence

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b1;
    b = 1'b0;
    @(posedge clk); // s starts (a=1)
    a = 1'b0;
    @(posedge clk); // s.triggered should be true here
    @(posedge clk);
    $finish;
  end

  assert property (@(posedge clk) not s.triggered);
endmodule

// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=60000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Runtime semantics: sequence `.triggered` should feed assertion truth.
// With both `a` and `b` low, `s.triggered` is false at sampled edges.

module top;
  reg clk;
  reg a, b;

  sequence s;
    @(posedge clk) a ##1 b;
  endsequence

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b0;
    b = 1'b0;
    @(posedge clk);
    @(posedge clk);
    @(posedge clk);
    $finish;
  end

  assert property (@(posedge clk) s.triggered);
endmodule

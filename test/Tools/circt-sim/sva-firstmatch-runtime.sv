// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=60000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Runtime semantics: first_match should preserve a definite failing sequence.
// With `a` low at sampled edges, `first_match(a ##[0:1] b)` cannot match.

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

  assert property (@(posedge clk) first_match(a ##[0:1] b));
endmodule

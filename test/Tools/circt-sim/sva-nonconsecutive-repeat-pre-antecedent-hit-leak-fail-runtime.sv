// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=200000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Runtime semantics: pre-antecedent hits of b must not count toward the
// consequent repetition in a |-> b[=2].

module top;
  reg clk;
  reg a, b;

  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b0;
    b = 1'b0;

    // Pre-antecedent hit; must not contribute to post-antecedent count.
    @(posedge clk); // cycle 1
    b = 1'b1;

    @(posedge clk); // cycle 2
    b = 1'b0;

    // Antecedent fires.
    @(posedge clk); // cycle 3
    a = 1'b1;

    @(posedge clk); // cycle 4
    a = 1'b0;

    // Only one post-antecedent hit.
    @(posedge clk); // cycle 5
    b = 1'b1;

    @(posedge clk); // cycle 6
    b = 1'b0;

    @(posedge clk); // cycle 7
    @(posedge clk); // cycle 8
    $finish;
  end

  assert property (@(posedge clk) a |-> b[=2]);
endmodule

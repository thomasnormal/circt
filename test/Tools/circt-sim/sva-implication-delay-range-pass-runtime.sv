// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=80000000 2>&1 | FileCheck %s
// CHECK: SVA_PASS: bounded implication satisfied

// Runtime semantics: `a |-> ##[1:2] b` should pass when `b` matches at the
// latest allowed sample in the bounded consequent window.

module top;
  reg clk;
  reg a, b;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b0;
    b = 1'b0;

    @(posedge clk); // cycle 1
    a = 1'b1;       // antecedent will be sampled high at cycle 2
    b = 1'b0;

    @(posedge clk); // cycle 2
    a = 1'b0;
    b = 1'b0;       // first allowed consequent sample (##1): no match yet

    @(posedge clk); // cycle 3
    b = 1'b1;       // second allowed consequent sample (##2): match

    @(posedge clk); // cycle 4
    $display("SVA_PASS: bounded implication satisfied");
    $finish;
  end

  assert property (@(posedge clk) a |-> ##[1:2] b);
endmodule

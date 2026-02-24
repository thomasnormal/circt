// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s
// CHECK: SVA_PASS: reject_on preserved passing implication
// CHECK-NOT: SVA assertion failed

// Runtime semantics: reject_on(c) should preserve normal implication behavior
// when c never becomes true.

module top;
  reg clk;
  reg a, b, c;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b0;
    b = 1'b0;
    c = 1'b0;

    @(posedge clk); // cycle 1
    a = 1'b1;       // antecedent sampled at cycle 2
    b = 1'b1;       // consequent satisfied

    @(posedge clk); // cycle 2
    a = 1'b0;
    b = 1'b1;
    c = 1'b0;

    @(posedge clk); // cycle 3
    $display("SVA_PASS: reject_on preserved passing implication");
    $finish;
  end

  assert property (@(posedge clk) reject_on(c) (a |-> ##1 b));
endmodule

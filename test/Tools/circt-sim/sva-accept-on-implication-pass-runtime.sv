// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s
// CHECK: SVA_PASS: accept_on aborted pending implication
// CHECK-NOT: SVA assertion failed

// Runtime semantics: accept_on(c) must vacuously pass when c is sampled high
// while an implication consequent obligation is still pending.

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
    a = 1'b1;       // antecedent sampled high at cycle 2

    @(posedge clk); // cycle 2
    a = 1'b0;
    b = 1'b0;       // consequent still unsatisfied

    @(negedge clk); // make c visible at the next sampled edge
    c = 1'b1;

    @(posedge clk); // cycle 3: accept_on condition sampled high, aborts pending
    @(posedge clk); // cycle 4
    $display("SVA_PASS: accept_on aborted pending implication");
    $finish;
  end

  assert property (@(posedge clk) accept_on(c) (a |-> ##1 b));
endmodule

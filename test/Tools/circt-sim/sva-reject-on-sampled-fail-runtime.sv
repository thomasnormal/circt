// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Runtime semantics: reject_on(c) must force failure when c is sampled high,
// even if the implication consequent would otherwise hold.

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
    b = 1'b1;       // consequent would pass

    @(negedge clk);
    c = 1'b1;       // sampled high at next posedge

    @(posedge clk); // cycle 2: reject_on should force failure
    a = 1'b0;
    b = 1'b1;
    c = 1'b0;

    @(posedge clk); // cycle 3
    $finish;
  end

  assert property (@(posedge clk) reject_on(c) (a |-> ##1 b));
endmodule

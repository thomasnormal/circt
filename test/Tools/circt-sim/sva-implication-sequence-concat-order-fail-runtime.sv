// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=300000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Deterministic sampled-value test for ordered concat semantics in implication.
//
// The consequent is ((##[1:2] b) ##[1:2] c). With a single antecedent trigger,
// b and c are both sampled high at +1 only. This does not satisfy the required
// order (c must occur 1..2 cycles after b), so the property must fail.

module top;
  reg clk;
  reg a;
  reg b;
  reg c;

  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b1;
    b = 1'b0;
    c = 1'b0;

    // Program values on negedge so sampled posedge values are deterministic.
    @(negedge clk);
    a = 1'b0;
    b = 1'b1;
    c = 1'b1;

    @(negedge clk);
    b = 1'b0;
    c = 1'b0;

    @(negedge clk);
    b = 1'b0;
    c = 1'b0;

    @(negedge clk);
    b = 1'b0;
    c = 1'b0;

    @(negedge clk);
    b = 1'b0;
    c = 1'b0;

    repeat (3) @(posedge clk);
    $finish;
  end

  assert property (@(posedge clk) a |-> ((##[1:2] b) ##[1:2] c));
endmodule

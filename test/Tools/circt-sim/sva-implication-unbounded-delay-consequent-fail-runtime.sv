// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=120000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Regression: if antecedent triggers and `##[1:$] a` never matches, the
// implication must fail at simulation end.

module top;
  reg clk;
  reg start;
  reg a;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    start = 1'b1;
    a = 1'b0;
    @(posedge clk);
    start = 1'b0;
    repeat (3) @(posedge clk);
    $finish;
  end

  assert property (@(posedge clk) start |-> ##[1:$] a);
endmodule

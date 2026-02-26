// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=200000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time 45000000 fs
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Overlapping antecedents: the older window is satisfied, but the younger
// window has no valid first_match hit and must still fail.

module top;
  reg clk;
  reg a;
  reg b;

  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b0;
    b = 1'b0;

    @(posedge clk); // 5ns
    a = 1'b1;

    @(posedge clk); // 15ns antecedent #1 sampled
    b = 1'b1;

    @(posedge clk); // 25ns antecedent #2 sampled, b satisfies #1 at +1
    a = 1'b0;
    b = 1'b0;

    @(posedge clk); // 35ns: #2 earliest candidate (+1) misses
    @(posedge clk); // 45ns: #2 latest candidate (+2) misses -> fail

    $finish;
  end

  assert property (@(posedge clk) a |-> first_match(##[1:2] b));
endmodule

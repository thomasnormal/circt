// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time 15000000 fs
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Runtime semantics: for one-shot antecedents, `a s_until b` in an overlapped
// implication must fail as soon as `a` becomes false before `b` is observed.

module top;
  reg clk;
  reg start;
  reg a;
  reg b;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    start = 1'b0;
    a = 1'b0;
    b = 1'b0;
    #1 start = 1'b1;
    a = 1'b1;
    #8 start = 1'b0;
    #2 a = 1'b0; // t=11 -> sampled low at t=15 before any b
    #20 $finish;
  end

  assert property (@(posedge clk) start |-> (a s_until b));
endmodule

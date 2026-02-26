// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s
// CHECK-NOT: SVA assertion failed
// CHECK: Simulation completed

// Runtime semantics: for one-shot antecedents, `a s_until_with b` should pass
// when `a` holds through the overlap cycle where `b` is true.

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
    #3 b = 1'b1; // t=12 -> sampled high at t=15 with a still high
    #8 b = 1'b0;
    a = 1'b0;
    #20 $finish;
  end

  assert property (@(posedge clk) start |-> (a s_until_with b));
endmodule

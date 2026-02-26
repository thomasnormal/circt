// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=200000000 2>&1 | FileCheck %s
// CHECK: SVA_PASS_OR_RANGE
// CHECK-NOT: SVA assertion failed at time

// Bounded variable-length sequence consequents composed via `or` should pass
// when any branch matches within the consequent window.

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
    a = 1'b0;
    b = 1'b0;
    c = 1'b0;

    @(posedge clk); // 5ns
    a = 1'b1;

    @(posedge clk); // 15ns antecedent sampled
    a = 1'b0;
    b = 1'b1;

    @(posedge clk); // 25ns: b satisfies ##[1:2] branch
    b = 1'b0;

    repeat (3) @(posedge clk);

    $display("SVA_PASS_OR_RANGE");
    $finish;
  end

  assert property (@(posedge clk) a |-> (##[1:2] b or ##[3:4] c));
endmodule

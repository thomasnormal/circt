// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=200000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time 55000000 fs
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Bounded variable-length sequence consequents composed via `or` must be
// tracked as obligation windows in implication checking.
//
// Here `a` is true once, while both `b` and `c` stay low. The consequent
// `(##[1:2] b or ##[3:4] c)` has a bounded window [1:4], so the property must
// fail when that window closes.

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

    // Keep consequent inputs low through the entire [1:4] window.
    repeat (6) @(posedge clk);

    $finish;
  end

  assert property (@(posedge clk) a |-> (##[1:2] b or ##[3:4] c));
endmodule

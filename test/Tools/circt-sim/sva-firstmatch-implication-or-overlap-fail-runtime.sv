// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=250000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time 65000000 fs
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Overlapping antecedents with top-level first_match over OR'd bounded ranges.
// The older antecedent is satisfied by b at +1; the younger antecedent has no
// match in [1:4] and must fail when its window closes.

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

    @(posedge clk); // 15ns antecedent #1 sampled
    b = 1'b1;       // satisfies #1 at +1 (25ns)

    @(posedge clk); // 25ns antecedent #2 sampled
    a = 1'b0;
    b = 1'b0;

    // Keep b/c low so #2 cannot match in [1:4].
    repeat (5) @(posedge clk);

    $finish;
  end

  assert property (@(posedge clk) a |-> first_match(##[1:2] b or ##[3:4] c));
endmodule

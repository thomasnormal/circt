// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=120000000 2>&1 | FileCheck %s
// CHECK: SVA_PASS: implication with ##[1:$] consequent satisfied
// CHECK-NOT: SVA assertion failed
// CHECK-NOT: SVA assertion failure(s)

// Regression: antecedent triggers once, consequent `##[1:$] a` is satisfied
// later, and simulation ends on that satisfying sampled edge.

module top;
  reg clk;
  reg start;
  reg a;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    start = 1'b0;
    a = 1'b0;
    @(posedge clk);
    start = 1'b1;      // antecedent will trigger on next sampled edge
    @(posedge clk);
    start = 1'b0;
    @(posedge clk);
    a = 1'b1;          // becomes visible on following sampled edge
    @(posedge clk);
    $display("SVA_PASS: implication with ##[1:$] consequent satisfied");
    $finish;
  end

  assert property (@(posedge clk) $rose(start) |-> ##[1:$] a);
endmodule

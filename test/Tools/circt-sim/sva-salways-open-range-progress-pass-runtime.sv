// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=90000000 2>&1 | FileCheck %s
// CHECK-NOT: SVA assertion failed
// CHECK: Simulation completed

// Runtime semantics: strong open-range repetition passes when lower-bound
// progress is achieved and the predicate remains true.
// Equivalent to strong open-range always for this unary predicate pattern.

module top;
  reg clk;
  reg a;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b1;
    repeat (6) @(posedge clk);
    $finish;
  end

  assert property (@(posedge clk) strong(a[*2:$]));
endmodule

// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=50000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Runtime semantics: strong open-range repetition must fail if simulation ends
// before the lower bound can be reached.
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
    @(posedge clk);
    @(posedge clk);
    $finish;
  end

  assert property (@(posedge clk) strong(a[*3:$]));
endmodule

// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=90000000 2>&1 | FileCheck %s
// CHECK: SVA assumption failed at time
// CHECK: SVA assumption failure(s)
// CHECK: exit code 1

// Runtime semantics: weak open-range always in a clocked assume must fail if
// the predicate is violated after lower-bound progress.

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
    a = 1'b0;
    @(posedge clk);
    $finish;
  end

  assume property (@(posedge clk) always [1:$] a);
endmodule

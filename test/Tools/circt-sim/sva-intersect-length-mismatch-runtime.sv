// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Runtime semantics: intersect requires same start and end time. These
// sequences have fixed but different lengths, so they can never intersect.

module top;
  reg clk;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    repeat (4) @(posedge clk);
    $finish;
  end

  assert property (@(posedge clk) (##1 1'b1) intersect 1'b1);
endmodule

// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=50000000 2>&1 | FileCheck %s
// CHECK-NOT: SVA assumption failed
// CHECK: Simulation completed

// Runtime semantics: weak open-range always in a clocked assume is vacuously
// true when simulation ends before the lower bound is reached.

module top;
  reg clk;
  reg a;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b0;
    @(posedge clk);
    @(posedge clk);
    $finish;
  end

  assume property (@(posedge clk) always [3:$] a);
endmodule

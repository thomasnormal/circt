// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=90000000 2>&1 | FileCheck %s
// CHECK-NOT: SVA assumption failed
// CHECK: Simulation completed

// Runtime semantics: weak open-range always in a clocked assume remains
// vacuously true for a nested property operand if run length is below the
// lower bound.

module top;
  reg clk;
  reg a;
  reg b;

  property p;
    @(posedge clk) a |-> b;
  endproperty

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1;
    b = 0;
    @(posedge clk);
    @(posedge clk);
    $finish;
  end

  assume property (@(posedge clk) always [3:$] p);
endmodule

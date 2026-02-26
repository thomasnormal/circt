// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=90000000 2>&1 | FileCheck %s
// CHECK-NOT: SVA assertion failed
// CHECK: Simulation completed

// Runtime semantics: strong(b[->2]) passes when two hits are observed.

module top;
  reg clk;
  reg b;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    b = 1'b0;
    @(posedge clk);
    b = 1'b1;
    @(posedge clk);
    b = 1'b0;
    @(posedge clk);
    b = 1'b1;
    @(posedge clk);
    b = 1'b0;
    @(posedge clk);
    $finish;
  end

  assert property (@(posedge clk) strong(b[->2]));
endmodule

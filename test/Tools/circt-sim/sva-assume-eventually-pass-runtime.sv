// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=60000000 2>&1 | FileCheck %s
// CHECK-NOT: SVA assumption failed
// CHECK: Simulation completed

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
    a = 1'b1;
    @(posedge clk);
    @(posedge clk);
    $finish;
  end

  assume property (@(posedge clk) s_eventually a);
endmodule

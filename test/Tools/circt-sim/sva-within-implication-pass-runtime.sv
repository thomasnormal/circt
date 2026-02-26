// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s
// CHECK: SVA_PASS_WITHIN

module top;
  reg clk;
  reg t, a, b;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    t = 1'b1;
    a = 1'b1;
    b = 1'b1;
    repeat (4) @(posedge clk);
    $display("SVA_PASS_WITHIN");
    $finish;
  end

  assert property (@(posedge clk) t |-> (a within (##1 b)));
endmodule

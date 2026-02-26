// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=60000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Regression: clocked assertion runtime state must be per-instance.
// One failing and one passing instance of the same asserted module should fail.

module child(input logic clk, input logic a);
  assert property (@(posedge clk) a);
endmodule

module top;
  logic clk;
  logic bad;
  logic good;

  child u_bad(.clk(clk), .a(bad));
  child u_good(.clk(clk), .a(good));

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    bad = 1'b0;
    good = 1'b1;
    @(posedge clk);
    @(posedge clk);
    $finish;
  end
endmodule

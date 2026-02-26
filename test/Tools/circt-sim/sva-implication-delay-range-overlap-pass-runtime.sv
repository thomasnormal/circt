// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=120000000 2>&1 | FileCheck %s
// CHECK: SVA_PASS_OVERLAP

// Overlapping obligations: both antecedent samples should be satisfied by a
// single consequent sample if it falls in both bounded windows.

module top;
  reg clk;
  reg a, b;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b0;
    b = 1'b0;

    @(posedge clk); // cycle 1
    a = 1'b1;
    b = 1'b0;

    @(posedge clk); // cycle 2
    a = 1'b1;
    b = 1'b0;

    @(posedge clk); // cycle 3
    a = 1'b0;
    b = 1'b1;

    @(posedge clk); // cycle 4
    $display("SVA_PASS_OVERLAP");
    $finish;
  end

  assert property (@(posedge clk) a |-> ##[1:2] b);
endmodule

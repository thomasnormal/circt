// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=120000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Overlapping obligations: first antecedent is satisfied, second is not. The
// checker must still fail once the second bounded window closes.

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
    b = 1'b1;

    @(posedge clk); // cycle 3
    a = 1'b0;
    b = 1'b0;

    @(posedge clk); // cycle 4
    b = 1'b0;

    @(posedge clk); // cycle 5
    $finish;
  end

  assert property (@(posedge clk) a |-> ##[1:2] b);
endmodule

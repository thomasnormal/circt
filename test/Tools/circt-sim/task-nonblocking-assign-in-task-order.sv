// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// REQUIRES: slang
//
// Regression: nonblocking assignment inside a task must preserve NBA timing.
// A read in the same active-region timestep should see the old value.

module top;
  logic clk = 0;
  logic x = 0;
  always #5 clk = ~clk;

  task automatic setx();
    x <= 1'b1;
  endtask

  initial begin
    @(posedge clk);
    setx();
    $display("AFTER_CALL x=%0b", x);
    #1 $display("AFTER_1 x=%0b", x);
    $finish;
  end
endmodule

// CHECK: AFTER_CALL x=0
// CHECK-NEXT: AFTER_1 x=1

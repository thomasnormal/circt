// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// RUN: circt-sim %t.mlir --top tb 2>&1 | FileCheck %s
// Regression for issue #31: cross-variable constraints must not drop
// individual per-variable bounds.

module tb;
  class Mixed;
    rand bit [3:0] x;
    rand bit [3:0] y;
    constraint c { x < 4; y < 4; x + y < 6; }
  endclass

  Mixed m;
  int xViol = 0;
  int yViol = 0;
  int sumViol = 0;

  initial begin
    m = new;
    repeat (200) begin
      void'(m.randomize());
      if (m.x >= 4) xViol++;
      if (m.y >= 4) yViol++;
      if (m.x + m.y >= 6) sumViol++;
    end

    if (xViol == 0 && yViol == 0 && sumViol == 0)
      $display("PASS");
    else
      $display("FAIL x=%0d y=%0d sum=%0d", xViol, yViol, sumViol);
    $finish;
  end

  // CHECK: PASS
  // CHECK-NOT: FAIL
endmodule

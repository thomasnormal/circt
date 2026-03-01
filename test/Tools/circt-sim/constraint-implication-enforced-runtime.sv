// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// RUN: circt-sim %t.mlir --top tb 2>&1 | FileCheck %s
// Regression for issue #41: implication constraints must enforce consequents.

module tb;
  class A;
    rand bit flag;
    rand bit [7:0] val;
    constraint c { flag -> (val > 100); }
  endclass

  A obj;
  int fail = 0;

  initial begin
    obj = new;
    repeat (128) begin
      void'(obj.randomize());
      if (obj.flag && obj.val <= 100)
        fail++;
    end

    if (fail == 0)
      $display("PASS");
    else
      $display("FAIL fail=%0d", fail);
    $finish;
  end

  // CHECK: PASS
  // CHECK-NOT: FAIL
endmodule

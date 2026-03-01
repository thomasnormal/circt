// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// RUN: circt-sim %t.mlir --top tb 2>&1 | FileCheck %s
// Regression for issue #42: obj.rand_mode(0) disables randomization of all
// rand properties on the object.

module tb;
  class C;
    rand int x;
    rand int y;
    constraint c { x inside {[1:10]}; y inside {[1:10]}; }
  endclass

  C obj;
  int stayed = 0;
  int changed = 0;

  initial begin
    obj = new;
    obj.x = 55;
    obj.y = 44;

    obj.rand_mode(0);
    void'(obj.randomize());
    stayed = (obj.x == 55 && obj.y == 44);

    obj.rand_mode(1);
    void'(obj.randomize());
    changed = (obj.x != 55 || obj.y != 44);

    if (stayed && changed)
      $display("PASS");
    else
      $display("FAIL stayed=%0d changed=%0d x=%0d y=%0d",
               stayed, changed, obj.x, obj.y);
    $finish;
  end

  // CHECK: PASS
  // CHECK-NOT: FAIL
endmodule

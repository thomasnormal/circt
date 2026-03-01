// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// RUN: circt-sim %t.mlir --top tb 2>&1 | FileCheck %s
// Regression for issue #45: dist constraints must restrict generated values.

module tb;
  class WeightedPkt;
    rand bit [1:0] kind;
    constraint kind_c { kind dist {0:=1, 1:=3, 2:=6}; }
  endclass

  WeightedPkt p;
  int fail = 0;

  initial begin
    p = new;
    repeat (128) begin
      void'(p.randomize());
      if (!(p.kind inside {0, 1, 2}))
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

// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// RUN: circt-sim %t.mlir --top tb 2>&1 | FileCheck %s
// Regression for issue #38: value-list `inside {a,b,c}` must be enforced.

module tb;
  class C;
    rand bit [7:0] x;
    constraint c_list { x inside {10, 20, 30, 40}; }
  endclass

  C obj;
  int fail = 0;

  initial begin
    obj = new;
    repeat (64) begin
      void'(obj.randomize());
      if (!(obj.x inside {10, 20, 30, 40}))
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

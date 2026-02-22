// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test that cross-variable constraint x + y < 100 is enforced.
// Bug: getPropertyName() returns "" for AddOp, silently dropping constraint.
class sum_bounded;
  rand bit [7:0] x;
  rand bit [7:0] y;
  constraint c_sum { x + y < 100; }

  function new();
    x = 0;
    y = 0;
  endfunction
endclass

module top;
  initial begin
    sum_bounded obj = new();
    int ok;
    int all_valid = 1;
    int i;
    for (i = 0; i < 100; i = i + 1) begin
      ok = obj.randomize();
      if (obj.x + obj.y >= 100)
        all_valid = 0;
    end
    // CHECK: sum_constraint=1
    $display("sum_constraint=%0d", all_valid);
    $finish;
  end
endmodule

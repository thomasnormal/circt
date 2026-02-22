// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test that cross-variable inequality constraint x != y is enforced.
// Bug: NeOp not handled by tryExtractDynBound, constraint silently dropped.
class neq_pair;
  rand bit [7:0] x;
  rand bit [7:0] y;
  constraint c_neq { x != y; }

  function new();
    x = 0;
    y = 0;
  endfunction
endclass

module top;
  initial begin
    neq_pair obj = new();
    int ok;
    int all_diff = 1;
    int i;
    for (i = 0; i < 100; i = i + 1) begin
      ok = obj.randomize();
      if (obj.x == obj.y)
        all_diff = 0;
    end
    // CHECK: neq_constraint=1
    $display("neq_constraint=%0d", all_diff);
    $finish;
  end
endmodule

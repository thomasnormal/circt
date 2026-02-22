// RUN: circt-verilog %s --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test that cross-variable constraint solving is deterministic:
// same seed produces same results across multiple randomize() calls.
class det_test;
  rand bit [7:0] x;
  rand bit [7:0] y;
  constraint c { x + y < 100; }

  function new();
    x = 0;
    y = 0;
  endfunction
endclass

module top;
  initial begin
    det_test obj = new();
    int x1, y1, x2, y2;

    obj.srandom(42);
    void'(obj.randomize());
    x1 = obj.x;
    y1 = obj.y;

    obj.srandom(42);
    void'(obj.randomize());
    x2 = obj.x;
    y2 = obj.y;

    // CHECK: deterministic=1
    $display("deterministic=%0d", (x1 == x2) && (y1 == y2));
    $finish;
  end
endmodule

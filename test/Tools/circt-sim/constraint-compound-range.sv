// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include -o %t.mlir 2>&1 | FileCheck %s --check-prefix=VERILOG --allow-empty
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test: compound range constraints from inside {[a:b]} patterns.
// Slang compiles `inside {[a:b]}` into `and(uge(x, a), ule(x, b))`.
// The constraint extraction must decompose these and() expressions.

// VERILOG-NOT: error

class a;
    rand bit [7:0] x;
    constraint c { x >= 6; x <= 19; }
endclass

module top;
  initial begin
    a obj = new;
    obj.randomize();
    // CHECK: RANGE x_in_range=1
    $display("RANGE x_in_range=%0d", (obj.x >= 6 && obj.x <= 19) ? 1 : 0);
    $finish;
  end
endmodule

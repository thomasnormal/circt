// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include -o %t.mlir 2>&1 | FileCheck %s --check-prefix=VERILOG --allow-empty
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test: basic dist constraint application.
// IEEE 1800-2017 §18.5.4

// VERILOG-NOT: error

class a;
    rand int x;
    constraint c { x dist { 42 := 1 }; }
endclass

module top;
  initial begin
    a obj = new;
    obj.randomize();
    // With dist {42 := 1} as the only value, x must be 42
    // CHECK: DIST x=42
    $display("DIST x=%0d", obj.x);
    $finish;
  end
endmodule

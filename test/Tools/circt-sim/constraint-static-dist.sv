// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include -o %t.mlir 2>&1 | FileCheck %s --check-prefix=VERILOG --allow-empty
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test: static constraint block with dist.
// IEEE 1800-2017 §18.5.4, §18.5.1 (static constraints)

// VERILOG-NOT: error

class a;
    rand int x;
    static constraint c { x dist { 99 := 1 }; }
endclass

module top;
  initial begin
    a obj = new;
    obj.randomize();
    // With static dist {99 := 1} as the only value, x must be 99
    // CHECK: STATIC_DIST x=99
    $display("STATIC_DIST x=%0d", obj.x);
    $finish;
  end
endmodule

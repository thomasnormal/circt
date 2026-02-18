// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test constraint_mode() method
module top;
  class packet;
    rand int data;
    constraint c_small { data < 100; }
  endclass

  initial begin
    packet p = new();

    // constraint_mode should be 1 (enabled) by default
    // CHECK: constraint_on=1
    $display("constraint_on=%0d", p.c_small.constraint_mode());

    // Disable constraint
    p.c_small.constraint_mode(0);
    // CHECK: constraint_off=0
    $display("constraint_off=%0d", p.c_small.constraint_mode());

    // Re-enable
    p.c_small.constraint_mode(1);
    // CHECK: constraint_reenabled=1
    $display("constraint_reenabled=%0d", p.c_small.constraint_mode());

    $finish;
  end
endmodule

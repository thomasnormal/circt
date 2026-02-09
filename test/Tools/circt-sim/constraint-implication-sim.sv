// Test that implication constraints (b1 == 5 -> b2 == 10) are applied correctly
// during randomization. IEEE 1800-2017 Section 18.5.6.
// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

class a;
  rand int b1, b2;
  constraint c1 { b1 == 5; }
  constraint c2 { b1 == 5 -> b2 == 10; }
endclass

module top;
  initial begin
    a obj = new;
    obj.randomize();
    if (obj.b1 == 5 && obj.b2 == 10)
      $display("IMPLICATION SUCCESS b1=%0d b2=%0d", obj.b1, obj.b2);
    else
      $display("IMPLICATION FAILED b1=%0d b2=%0d", obj.b1, obj.b2);
    $finish;
  end
endmodule

// CHECK: IMPLICATION SUCCESS
// CHECK: Simulation completed

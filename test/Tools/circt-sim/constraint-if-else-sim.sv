// Test that if-else constraints with nested conditions are applied correctly.
// IEEE 1800-2017 Section 18.5.7.
// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

class a;
  rand int b1, b2, b3;
  constraint c1 { b1 == 5; }
  constraint c2 { b2 == 3; }
  constraint c3 { if (b1 == 5)
                    if (b2 == 2) b3 == 4;
                    else b3 == 10; }
endclass

module top;
  initial begin
    a obj = new;
    obj.randomize();
    // b1=5, b2=3 (not 2), so b3 should be 10 (else branch)
    if (obj.b3 == 10)
      $display("IF_ELSE SUCCESS b3=%0d", obj.b3);
    else
      $display("IF_ELSE FAILED b3=%0d", obj.b3);
    $finish;
  end
endmodule

// CHECK: IF_ELSE SUCCESS
// CHECK: Simulation completed

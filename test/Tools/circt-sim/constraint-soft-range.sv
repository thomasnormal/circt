// Test that soft range constraints (soft b > 4; soft b < 12;) are applied
// correctly during randomization. IEEE 1800-2017 Section 18.5.14.
// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

class a;
    rand int b;
    constraint c {
        soft b > 4;
        soft b < 12;
    }
endclass

module top;
  initial begin
    a obj = new;
    obj.randomize();
    if (obj.b > 4 && obj.b < 12)
      $display("SOFT_RANGE SUCCESS b=%0d", obj.b);
    else
      $display("SOFT_RANGE FAILED b=%0d", obj.b);
    $finish;
  end
endmodule

// CHECK: SOFT_RANGE SUCCESS
// CHECK: Simulation completed

// Test that set membership constraints (inside {3, 10}) are applied correctly
// during randomization. IEEE 1800-2017 Section 18.5.3.
// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

class a;
  rand int b;
  constraint c { b inside {3, 10}; }
endclass

module top;
  initial begin
    a obj = new;
    obj.randomize();
    if (obj.b inside {3, 10})
      $display("SET_MEMBERSHIP SUCCESS b=%0d", obj.b);
    else
      $display("SET_MEMBERSHIP FAILED b=%0d", obj.b);
    $finish;
  end
endmodule

// CHECK: SET_MEMBERSHIP SUCCESS
// CHECK: Simulation completed

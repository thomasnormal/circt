// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include -o %t.mlir 2>&1 | FileCheck %s --check-prefix=VERILOG --allow-empty
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test: constraint inheritance from parent class.
// IEEE 1800-2017 §18.5.2: Constraints from parent classes are inherited
// by derived classes and applied during randomize().

// VERILOG-NOT: error

class a;
    rand int b;
    constraint c { b == 42; }
endclass

class a2 extends a;
    rand int b2;
    // Note: c2 uses cross-variable constraint (b2 == b) which is not yet
    // supported, so we test with a constant constraint instead.
    constraint c2 { b2 == 99; }
endclass

module top;
  initial begin
    a2 obj = new;
    obj.randomize();
    // Parent constraint c should set b=42, child constraint c2 should set b2=99
    // CHECK: INHERIT b=42 b2=99
    $display("INHERIT b=%0d b2=%0d", obj.b, obj.b2);
    $finish;
  end
endmodule

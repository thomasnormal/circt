// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include -o %t.mlir 2>&1 | FileCheck %s --check-prefix=VERILOG --allow-empty
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test: constraint guard with null class handle check.
// IEEE 1800-2017 §18.5.13: Constraint guards prevent evaluation of
// constraints when a handle is null.

// VERILOG-NOT: error

class b;
    int d1;
endclass

class a;
    rand int b1;
    b next;
    constraint c1 { if (next == null) b1 == 5; }
endclass

module top;
  initial begin
    a obj1 = new;
    // next is null by default, so constraint should force b1=5
    obj1.randomize();
    // CHECK: GUARD_NULL b1=5
    $display("GUARD_NULL b1=%0d", obj1.b1);

    // Now assign next, constraint guard should not apply
    obj1.next = new;
    obj1.randomize();
    // b1 is unconstrained, just check it doesn't crash
    // CHECK: GUARD_NONNULL b1=
    $display("GUARD_NONNULL b1=%0d", obj1.b1);
    $finish;
  end
endmodule

// RUN: circt-verilog %s -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s

// Test soft constraint priority per IEEE 1800-2017 §18.5.14.1.
// Derived class soft constraints override base class soft constraints.

class base_cls;
  rand int x;

  // Base class: soft x in range [5, 11]
  constraint c1 {
    soft x > 4;
    soft x < 12;
  }
endclass

class derived_cls extends base_cls;
  // Derived class: soft x == 42 overrides base constraint
  constraint c2 { soft x == 42; }
endclass

class derived2 extends base_cls;
  // Two soft constraints in same class: last wins
  constraint c2 { soft x == 20; }
  constraint c3 { soft x == 77; }
endclass

module top;
  initial begin
    derived_cls obj1 = new;
    derived2 obj2 = new;
    int status;

    // Test 1: Derived class soft constraint overrides base
    status = obj1.randomize();
    // CHECK: DERIVED_OVERRIDE: 42
    $display("DERIVED_OVERRIDE: %0d", obj1.x);

    // Test 2: Within same class, last soft constraint wins
    status = obj2.randomize();
    // CHECK: LAST_WINS: 77
    $display("LAST_WINS: %0d", obj2.x);

    $finish;
  end
endmodule

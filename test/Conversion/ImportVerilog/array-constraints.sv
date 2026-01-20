// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test array constraint support for randomization
// IEEE 1800-2017 Section 18.5.5 "Uniqueness constraints"
// IEEE 1800-2017 Section 18.5.8 "Foreach constraints"

// Test class with unique array constraint
class test_unique_constraint;
  rand bit [7:0] arr[4];

  // Unique constraint - all elements must be different
  constraint c_unique {
    unique {arr};
  }

  function new();
  endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @test_unique_constraint
// CHECK: moore.class.propertydecl @arr : !moore.uarray<4 x i8>
// CHECK-SAME: rand_mode rand
// CHECK: moore.constraint.block @c_unique
// CHECK: moore.constraint.unique

// Test class with unique scalars constraint
class test_unique_scalars;
  rand int a, b, c;

  // Unique constraint on multiple scalars
  constraint c_unique {
    unique {a, b, c};
  }

  function new();
  endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @test_unique_scalars
// CHECK: moore.constraint.block @c_unique
// CHECK: moore.constraint.unique

// Test foreach with range constraint
class test_foreach_range;
  rand int arr[8];

  // Each element must be in range [0, 100]
  constraint c_range {
    foreach (arr[i]) {
      arr[i] inside {[0:100]};
    }
  }

  function new();
  endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @test_foreach_range
// CHECK: moore.constraint.block @c_range
// CHECK: moore.constraint.foreach

// Test foreach with comparison constraint
class test_foreach_compare;
  rand bit [7:0] data[4];

  // Each element must be less than 50
  constraint c_less {
    foreach (data[i]) {
      data[i] < 50;
    }
  }

  function new();
  endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @test_foreach_compare
// CHECK: moore.constraint.block @c_less
// CHECK: moore.constraint.foreach

// Test combined constraints
class test_combined_constraints;
  rand bit [7:0] arr[5];

  // Combined: unique elements in a specific range
  constraint c_unique { unique {arr}; }
  constraint c_range {
    foreach (arr[i]) {
      arr[i] inside {[1:100]};
    }
  }

  function new();
  endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @test_combined_constraints
// CHECK: moore.constraint.block @c_unique
// CHECK: moore.constraint.unique
// CHECK: moore.constraint.block @c_range
// CHECK: moore.constraint.foreach

// Test module instantiation
module top;
  initial begin
    automatic test_unique_constraint t1 = new();
    automatic test_unique_scalars t2 = new();
    automatic test_foreach_range t3 = new();
    automatic test_foreach_compare t4 = new();
    automatic test_combined_constraints t5 = new();

    void'(t1.randomize());
    void'(t2.randomize());
    void'(t3.randomize());
    void'(t4.randomize());
    void'(t5.randomize());
  end
endmodule

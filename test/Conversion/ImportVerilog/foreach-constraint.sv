// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test foreach constraint support for randomization
class test_foreach_constraint;
  rand bit [7:0] arr[4];
  rand int values[8];

  // Simple foreach constraint - ensure each element is non-zero
  constraint c_nonzero {
    foreach (arr[i]) {
      arr[i] != 0;
    }
  }

  // Foreach constraint with range constraint
  constraint c_range {
    foreach (values[j]) {
      values[j] inside {[1:100]};
    }
  }

  function new();
  endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @test_foreach_constraint
// CHECK: moore.class.propertydecl @arr : !moore.uarray<4 x i8>
// CHECK-SAME: rand_mode rand
// CHECK: moore.class.propertydecl @values : !moore.uarray<8 x i32>
// CHECK-SAME: rand_mode rand

// CHECK: moore.constraint.block @c_nonzero
// CHECK: moore.constraint.foreach %{{.*}} : !moore.uarray<4 x i8>
// CHECK-NEXT: ^bb0(%{{.*}}: !moore.i32):
// CHECK: moore.constraint.expr

// CHECK: moore.constraint.block @c_range
// CHECK: moore.constraint.foreach %{{.*}} : !moore.uarray<8 x i32>
// CHECK-NEXT: ^bb0(%{{.*}}: !moore.i32):
// CHECK: moore.constraint.expr

// Test class with multi-dimensional foreach
class test_multidim_foreach;
  rand bit [3:0] matrix[2][3];

  constraint c_matrix {
    foreach (matrix[i, j]) {
      matrix[i][j] < 10;
    }
  }

  function new();
  endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @test_multidim_foreach
// CHECK: moore.constraint.block @c_matrix
// CHECK: moore.constraint.foreach %{{.*}} : !moore.uarray<2 x uarray<3 x i4>>
// CHECK-NEXT: ^bb0(%{{.*}}: !moore.i32, %{{.*}}: !moore.i32):
// CHECK: moore.constraint.expr

// Test foreach with implication
class test_foreach_implication;
  rand bit mode;
  rand int data[4];

  constraint c_impl {
    foreach (data[i]) {
      mode -> data[i] > 0;
    }
  }

  function new();
  endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @test_foreach_implication
// CHECK: moore.constraint.block @c_impl
// CHECK: moore.constraint.foreach %{{.*}} : !moore.uarray<4 x i32>
// CHECK-NEXT: ^bb0(%{{.*}}: !moore.i32):
// CHECK: moore.constraint.implication

// Test foreach with queue (common in UVM)
class test_foreach_queue;
  rand int region_start;
  rand int region_end;
  int used_regions[$];

  // Constraint that ensures no overlap with used regions
  // (simplified version of UVM uvm_mem_mam_policy constraint)
  constraint c_no_overlap {
    foreach (used_regions[i]) {
      region_end < used_regions[i] || region_start > used_regions[i];
    }
  }

  function new();
  endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @test_foreach_queue
// CHECK: moore.constraint.block @c_no_overlap
// CHECK: moore.constraint.foreach %{{.*}} : !moore.queue<i32, 0>
// CHECK-NEXT: ^bb0(%{{.*}}: !moore.i32):
// CHECK: moore.constraint.expr

module top;
  initial begin
    automatic test_foreach_constraint t1 = new();
    automatic test_multidim_foreach t2 = new();
    automatic test_foreach_implication t3 = new();
    automatic test_foreach_queue t4 = new();
    void'(t1.randomize());
    void'(t2.randomize());
    void'(t3.randomize());
    void'(t4.randomize());
  end
endmodule

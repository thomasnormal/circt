// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// Foreach Implication Constraint Lowering Tests
// IEEE 1800-2017 Section 18.5.8 "foreach constraints"
// IEEE 1800-2017 Section 18.5.6 "Implication constraints"
//===----------------------------------------------------------------------===//
//
// This test file verifies that implication constraints (expr -> constraint)
// and if-else constraints (if (cond) constraint; else constraint;) work
// correctly within foreach loop contexts.
//
// Pattern 1: foreach (arr[i]) arr[i] -> constraint;
//   - Array element used as implication antecedent
//
// Pattern 2: foreach (arr[i]) if (cond) constraint;
//   - If-else constraint within foreach body
//
// The lowering strategy:
// - Both ConstraintForeachOp and ConstraintImplicationOp/ConstraintIfElseOp
//   are erased during MooreToCore conversion
// - Runtime validation happens via __moore_constraint_foreach_validate()
//   and __moore_randomize_basic() calls
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Test 1: Array element as implication condition
// Pattern: foreach (arr[i]) arr[i] -> constraint;
//===----------------------------------------------------------------------===//

/// SystemVerilog equivalent:
///   class ArrayElementImplication;
///     rand bit enable[4];
///     rand int data[4];
///     constraint c {
///       foreach (enable[i]) {
///         enable[i] -> (data[i] > 0);
///       }
///     }
///   endclass
///
/// Semantics: For each element, if enable[i] is true, then data[i] must be > 0

moore.class.classdecl @ArrayElementImplication {
  moore.class.propertydecl @enable : !moore.uarray<4 x i1> rand_mode rand
  moore.class.propertydecl @data : !moore.uarray<4 x i32> rand_mode rand
  moore.constraint.block @c_elem_cond {
  ^bb0(%enable: !moore.uarray<4 x i1>, %data: !moore.uarray<4 x i32>):
    moore.constraint.foreach %enable : !moore.uarray<4 x i1> {
    ^bb0(%i: !moore.i32):
      // Extract enable[i] and use as condition
      %en = moore.dyn_extract %enable from %i : !moore.uarray<4 x i1>, !moore.i32 -> !moore.i1
      moore.constraint.implication %en : i1 {
        // When enable[i] is true, data[i] must be positive
        %d = moore.dyn_extract %data from %i : !moore.uarray<4 x i32>, !moore.i32 -> !moore.i32
        %c0 = moore.constant 0 : i32
        %positive = moore.sgt %d, %c0 : i32 -> i1
        moore.constraint.expr %positive : i1
      }
    }
  }
}

// CHECK-LABEL: func.func @test_array_element_implication
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_array_element_implication(%obj: !moore.class<@ArrayElementImplication>) -> i1 {
  // Verify constraint ops are erased
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: moore.constraint.foreach
  // CHECK-NOT: moore.constraint.implication
  %success = moore.randomize %obj : !moore.class<@ArrayElementImplication>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test 2: If-else constraint within foreach
// Pattern: foreach (arr[i]) if (cond) constraint; else constraint;
//===----------------------------------------------------------------------===//

/// SystemVerilog equivalent:
///   class ForeachIfElseConstraint;
///     rand bit signed_mode;
///     rand int values[8];
///     constraint c {
///       foreach (values[i]) {
///         if (signed_mode)
///           values[i] inside {[-100:100]};
///         else
///           values[i] inside {[0:200]};
///       }
///     }
///   endclass

moore.class.classdecl @ForeachIfElseConstraint {
  moore.class.propertydecl @signed_mode : !moore.i1 rand_mode rand
  moore.class.propertydecl @values : !moore.uarray<8 x i32> rand_mode rand
  moore.constraint.block @c_ifelse_foreach {
  ^bb0(%signed_mode: !moore.i1, %values: !moore.uarray<8 x i32>):
    moore.constraint.foreach %values : !moore.uarray<8 x i32> {
    ^bb0(%i: !moore.i32):
      %v = moore.dyn_extract %values from %i : !moore.uarray<8 x i32>, !moore.i32 -> !moore.i32
      moore.constraint.if_else %signed_mode : i1 {
        // Signed mode: [-100, 100]
        %n100 = moore.constant -100 : i32
        %p100 = moore.constant 100 : i32
        %ge = moore.sge %v, %n100 : i32 -> i1
        %le = moore.sle %v, %p100 : i32 -> i1
        %in_range = moore.and %ge, %le : !moore.i1
        moore.constraint.expr %in_range : i1
      } else {
        // Unsigned mode: [0, 200]
        %c0 = moore.constant 0 : i32
        %c200 = moore.constant 200 : i32
        %ge = moore.sge %v, %c0 : i32 -> i1
        %le = moore.sle %v, %c200 : i32 -> i1
        %in_range = moore.and %ge, %le : !moore.i1
        moore.constraint.expr %in_range : i1
      }
    }
  }
}

// CHECK-LABEL: func.func @test_foreach_if_else
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_foreach_if_else(%obj: !moore.class<@ForeachIfElseConstraint>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: moore.constraint.foreach
  // CHECK-NOT: moore.constraint.if_else
  %success = moore.randomize %obj : !moore.class<@ForeachIfElseConstraint>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test 3: If-only constraint within foreach (no else branch)
// Pattern: foreach (arr[i]) if (cond) constraint;
//===----------------------------------------------------------------------===//

/// SystemVerilog equivalent:
///   class ForeachIfOnly;
///     rand bit enable;
///     rand int arr[4];
///     constraint c {
///       foreach (arr[i]) {
///         if (enable) arr[i] != 0;
///       }
///     }
///   endclass

moore.class.classdecl @ForeachIfOnlyConstraint {
  moore.class.propertydecl @enable : !moore.i1 rand_mode rand
  moore.class.propertydecl @arr : !moore.uarray<4 x i32> rand_mode rand
  moore.constraint.block @c_if_only {
  ^bb0(%enable: !moore.i1, %arr: !moore.uarray<4 x i32>):
    moore.constraint.foreach %arr : !moore.uarray<4 x i32> {
    ^bb0(%i: !moore.i32):
      moore.constraint.if_else %enable : i1 {
        %elem = moore.dyn_extract %arr from %i : !moore.uarray<4 x i32>, !moore.i32 -> !moore.i32
        %c0 = moore.constant 0 : i32
        %neq = moore.ne %elem, %c0 : i32 -> i1
        moore.constraint.expr %neq : i1
      }
    }
  }
}

// CHECK-LABEL: func.func @test_foreach_if_only
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_foreach_if_only(%obj: !moore.class<@ForeachIfOnlyConstraint>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: moore.constraint.foreach
  // CHECK-NOT: moore.constraint.if_else
  %success = moore.randomize %obj : !moore.class<@ForeachIfOnlyConstraint>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test 4: Index-based implication within foreach
// Pattern: foreach (arr[i]) (i < N) -> constraint;
//===----------------------------------------------------------------------===//

/// SystemVerilog equivalent:
///   class IndexImplication;
///     rand int data[8];
///     constraint c {
///       foreach (data[i]) {
///         (i < 4) -> (data[i] < 100);
///         (i >= 4) -> (data[i] >= 100);
///       }
///     }
///   endclass

moore.class.classdecl @IndexImplication {
  moore.class.propertydecl @data : !moore.uarray<8 x i32> rand_mode rand
  moore.constraint.block @c_idx {
  ^bb0(%data: !moore.uarray<8 x i32>):
    moore.constraint.foreach %data : !moore.uarray<8 x i32> {
    ^bb0(%i: !moore.i32):
      %c4 = moore.constant 4 : i32
      %c100 = moore.constant 100 : i32
      %i_lt_4 = moore.slt %i, %c4 : i32 -> i1
      %i_ge_4 = moore.sge %i, %c4 : i32 -> i1
      %elem = moore.dyn_extract %data from %i : !moore.uarray<8 x i32>, !moore.i32 -> !moore.i32

      // Lower half of array: data[i] < 100
      moore.constraint.implication %i_lt_4 : i1 {
        %lt100 = moore.slt %elem, %c100 : i32 -> i1
        moore.constraint.expr %lt100 : i1
      }
      // Upper half of array: data[i] >= 100
      moore.constraint.implication %i_ge_4 : i1 {
        %ge100 = moore.sge %elem, %c100 : i32 -> i1
        moore.constraint.expr %ge100 : i1
      }
    }
  }
}

// CHECK-LABEL: func.func @test_index_implication
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_index_implication(%obj: !moore.class<@IndexImplication>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: moore.constraint.foreach
  // CHECK-NOT: moore.constraint.implication
  %success = moore.randomize %obj : !moore.class<@IndexImplication>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test 5: Nested foreach with implication
// Pattern: foreach (a[i]) foreach (b[j]) (i == j) -> constraint;
//===----------------------------------------------------------------------===//

/// SystemVerilog equivalent:
///   class NestedForeachImpl;
///     rand int a[4];
///     rand int b[4];
///     constraint c {
///       foreach (a[i])
///         foreach (b[j])
///           (i == j) -> (a[i] == b[j]);
///     }
///   endclass

moore.class.classdecl @NestedForeachImpl {
  moore.class.propertydecl @a : !moore.uarray<4 x i32> rand_mode rand
  moore.class.propertydecl @b : !moore.uarray<4 x i32> rand_mode rand
  moore.constraint.block @c_nested {
  ^bb0(%a: !moore.uarray<4 x i32>, %b: !moore.uarray<4 x i32>):
    moore.constraint.foreach %a : !moore.uarray<4 x i32> {
    ^bb0(%i: !moore.i32):
      moore.constraint.foreach %b : !moore.uarray<4 x i32> {
      ^bb0(%j: !moore.i32):
        %same_idx = moore.eq %i, %j : i32 -> i1
        moore.constraint.implication %same_idx : i1 {
          %ai = moore.dyn_extract %a from %i : !moore.uarray<4 x i32>, !moore.i32 -> !moore.i32
          %bj = moore.dyn_extract %b from %j : !moore.uarray<4 x i32>, !moore.i32 -> !moore.i32
          %eq = moore.eq %ai, %bj : i32 -> i1
          moore.constraint.expr %eq : i1
        }
      }
    }
  }
}

// CHECK-LABEL: func.func @test_nested_foreach_impl
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_nested_foreach_impl(%obj: !moore.class<@NestedForeachImpl>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: moore.constraint.foreach
  // CHECK-NOT: moore.constraint.implication
  %success = moore.randomize %obj : !moore.class<@NestedForeachImpl>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test 6: Complex packet-style constraint pattern
// Pattern: foreach with multiple conditional constraints
//===----------------------------------------------------------------------===//

/// SystemVerilog equivalent (UVM-style):
///   class Packet;
///     rand bit [1:0] burst_type;
///     rand bit [7:0] burst_data[4];
///     constraint c {
///       foreach (burst_data[i]) {
///         (burst_type == 0) -> (burst_data[i] == 0);
///         (burst_type == 1) -> (burst_data[i] inside {[1:127]});
///         (burst_type == 2) -> (burst_data[i] inside {[128:255]});
///       }
///     }
///   endclass

moore.class.classdecl @PacketConstraint {
  moore.class.propertydecl @burst_type : !moore.i2 rand_mode rand
  moore.class.propertydecl @burst_data : !moore.uarray<4 x i8> rand_mode rand
  moore.constraint.block @c_packet {
  ^bb0(%burst_type: !moore.i2, %burst_data: !moore.uarray<4 x i8>):
    moore.constraint.foreach %burst_data : !moore.uarray<4 x i8> {
    ^bb0(%i: !moore.i32):
      %elem = moore.dyn_extract %burst_data from %i : !moore.uarray<4 x i8>, !moore.i32 -> !moore.i8
      %c0_2 = moore.constant 0 : i2
      %c1_2 = moore.constant 1 : i2
      %c2_2 = moore.constant 2 : i2
      %is_type0 = moore.eq %burst_type, %c0_2 : i2 -> i1
      %is_type1 = moore.eq %burst_type, %c1_2 : i2 -> i1
      %is_type2 = moore.eq %burst_type, %c2_2 : i2 -> i1

      // Type 0: zero data
      moore.constraint.implication %is_type0 : i1 {
        %c0_8 = moore.constant 0 : i8
        %eq_zero = moore.eq %elem, %c0_8 : i8 -> i1
        moore.constraint.expr %eq_zero : i1
      }
      // Type 1: low range [1, 127]
      moore.constraint.implication %is_type1 : i1 {
        %c1_8 = moore.constant 1 : i8
        %c127 = moore.constant 127 : i8
        %ge1 = moore.uge %elem, %c1_8 : i8 -> i1
        %le127 = moore.ule %elem, %c127 : i8 -> i1
        %low_range = moore.and %ge1, %le127 : !moore.i1
        moore.constraint.expr %low_range : i1
      }
      // Type 2: high range [128, 255]
      moore.constraint.implication %is_type2 : i1 {
        %c128 = moore.constant 128 : i8
        %c255 = moore.constant 255 : i8
        %ge128 = moore.uge %elem, %c128 : i8 -> i1
        %le255 = moore.ule %elem, %c255 : i8 -> i1
        %high_range = moore.and %ge128, %le255 : !moore.i1
        moore.constraint.expr %high_range : i1
      }
    }
  }
}

// CHECK-LABEL: func.func @test_packet_constraint
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_packet_constraint(%obj: !moore.class<@PacketConstraint>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: moore.constraint.foreach
  // CHECK-NOT: moore.constraint.implication
  %success = moore.randomize %obj : !moore.class<@PacketConstraint>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test 7: Soft constraints within foreach implication
//===----------------------------------------------------------------------===//

/// SystemVerilog equivalent:
///   class SoftForeachImpl;
///     rand bit mode;
///     rand int arr[4];
///     constraint c {
///       foreach (arr[i]) {
///         mode -> soft arr[i] == 100;
///       }
///     }
///   endclass

moore.class.classdecl @SoftForeachImpl {
  moore.class.propertydecl @mode : !moore.i1 rand_mode rand
  moore.class.propertydecl @arr : !moore.uarray<4 x i32> rand_mode rand
  moore.constraint.block @c_soft {
  ^bb0(%mode: !moore.i1, %arr: !moore.uarray<4 x i32>):
    moore.constraint.foreach %arr : !moore.uarray<4 x i32> {
    ^bb0(%i: !moore.i32):
      moore.constraint.implication %mode : i1 {
        %elem = moore.dyn_extract %arr from %i : !moore.uarray<4 x i32>, !moore.i32 -> !moore.i32
        %c100 = moore.constant 100 : i32
        %eq100 = moore.eq %elem, %c100 : i32 -> i1
        moore.constraint.expr %eq100 : i1 soft
      }
    }
  }
}

// CHECK-LABEL: func.func @test_soft_foreach_impl
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_soft_foreach_impl(%obj: !moore.class<@SoftForeachImpl>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: moore.constraint.foreach
  // CHECK-NOT: moore.constraint.implication
  // CHECK-NOT: moore.constraint.expr
  %success = moore.randomize %obj : !moore.class<@SoftForeachImpl>
  return %success : i1
}

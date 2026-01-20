// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// Array Foreach Constraint Lowering Tests
//===----------------------------------------------------------------------===//
//
// Foreach constraints are processed during RandomizeOp conversion.
// The foreach op itself is erased after constraint extraction.
// Runtime validation uses __moore_constraint_foreach_validate().

/// Test class with simple foreach constraint on static array.
/// Corresponds to SystemVerilog:
///   class test;
///     rand bit [7:0] arr[4];
///     constraint c { foreach (arr[i]) { arr[i] != 0; } }
///   endclass
///
/// The foreach constraint is erased during MooreToCore lowering.
/// Validation happens via runtime during randomize().

moore.class.classdecl @ForeachStaticArray {
  moore.class.propertydecl @arr : !moore.uarray<4 x i8> rand_mode rand
  moore.constraint.block @c_nonzero {
  ^bb0(%arr: !moore.uarray<4 x i8>):
    moore.constraint.foreach %arr : !moore.uarray<4 x i8> {
    ^bb0(%i: !moore.i32):
      // Constraint body - validates each element is non-zero
      %c0 = moore.constant 0 : i8
      %elem = moore.dyn_extract %arr from %i : !moore.uarray<4 x i8>, !moore.i32 -> !moore.i8
      %neq = moore.ne %elem, %c0 : i8 -> i1
      moore.constraint.expr %neq : i1
    }
  }
}

// CHECK-LABEL: func.func @test_foreach_static
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_foreach_static(%obj: !moore.class<@ForeachStaticArray>) -> i1 {
  // The foreach constraint is erased; randomize proceeds with basic randomization
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: moore.constraint.foreach
  %success = moore.randomize %obj : !moore.class<@ForeachStaticArray>
  return %success : i1
}

/// Test foreach constraint with range validation.
/// Corresponds to SystemVerilog:
///   foreach (values[j]) { values[j] inside {[1:100]}; }

moore.class.classdecl @ForeachRangeConstraint {
  moore.class.propertydecl @values : !moore.uarray<8 x i32> rand_mode rand
  moore.constraint.block @c_range {
  ^bb0(%values: !moore.uarray<8 x i32>):
    moore.constraint.foreach %values : !moore.uarray<8 x i32> {
    ^bb0(%j: !moore.i32):
      %elem = moore.dyn_extract %values from %j : !moore.uarray<8 x i32>, !moore.i32 -> !moore.i32
      // Validate element is in range [1, 100]
      %c1 = moore.constant 1 : i32
      %c100 = moore.constant 100 : i32
      %ge = moore.sge %elem, %c1 : i32 -> i1
      %le = moore.sle %elem, %c100 : i32 -> i1
      %in_range = moore.and %ge, %le : !moore.i1
      moore.constraint.expr %in_range : i1
    }
  }
}

// CHECK-LABEL: func.func @test_foreach_range
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_foreach_range(%obj: !moore.class<@ForeachRangeConstraint>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: moore.constraint.foreach
  %success = moore.randomize %obj : !moore.class<@ForeachRangeConstraint>
  return %success : i1
}

/// Test foreach constraint on queue (dynamic array).
/// Corresponds to SystemVerilog:
///   int queue[$];
///   constraint c { foreach (queue[i]) { queue[i] > 0; } }

moore.class.classdecl @ForeachQueueConstraint {
  moore.class.propertydecl @queue : !moore.queue<i32, 0> rand_mode rand
  moore.constraint.block @c_positive {
  ^bb0(%queue: !moore.queue<i32, 0>):
    moore.constraint.foreach %queue : !moore.queue<i32, 0> {
    ^bb0(%i: !moore.i32):
      %elem = moore.dyn_extract %queue from %i : !moore.queue<i32, 0>, !moore.i32 -> !moore.i32
      %c0 = moore.constant 0 : i32
      %gt = moore.sgt %elem, %c0 : i32 -> i1
      moore.constraint.expr %gt : i1
    }
  }
}

// CHECK-LABEL: func.func @test_foreach_queue
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_foreach_queue(%obj: !moore.class<@ForeachQueueConstraint>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: moore.constraint.foreach
  %success = moore.randomize %obj : !moore.class<@ForeachQueueConstraint>
  return %success : i1
}

/// Test standalone foreach constraint (not in class context).
/// Standalone foreach ops are erased since validation happens at randomization time.

// CHECK-LABEL: func.func @test_standalone_foreach
// CHECK-SAME: (%[[ARR:.*]]: !hw.array<8xi32>)
func.func @test_standalone_foreach(%arr: !moore.uarray<8 x i32>) {
  // Standalone foreach is erased
  // CHECK-NOT: moore.constraint.foreach
  // CHECK: return
  moore.constraint.foreach %arr : !moore.uarray<8 x i32> {
  ^bb0(%i: !moore.i32):
    %c0 = moore.constant 0 : i32
    %neq = moore.ne %i, %c0 : i32 -> i1
    moore.constraint.expr %neq : i1
  }
  return
}

/// Test empty foreach body (should be erased).

// CHECK-LABEL: func.func @test_empty_foreach
// CHECK-SAME: (%[[ARR:.*]]: !hw.array<4xi8>)
func.func @test_empty_foreach(%arr: !moore.uarray<4 x i8>) {
  // Empty body should just be erased
  // CHECK-NOT: moore.constraint.foreach
  moore.constraint.foreach %arr : !moore.uarray<4 x i8> {
  }
  // CHECK: return
  return
}

/// Test foreach with implication inside.
/// Corresponds to SystemVerilog:
///   foreach (data[i]) { mode -> data[i] > 0; }

moore.class.classdecl @ForeachImplication {
  moore.class.propertydecl @mode : !moore.i1 rand_mode rand
  moore.class.propertydecl @data : !moore.uarray<4 x i32> rand_mode rand
  moore.constraint.block @c_impl {
  ^bb0(%mode: !moore.i1, %data: !moore.uarray<4 x i32>):
    moore.constraint.foreach %data : !moore.uarray<4 x i32> {
    ^bb0(%i: !moore.i32):
      moore.constraint.implication %mode : i1 {
        %elem = moore.dyn_extract %data from %i : !moore.uarray<4 x i32>, !moore.i32 -> !moore.i32
        %c0 = moore.constant 0 : i32
        %gt = moore.sgt %elem, %c0 : i32 -> i1
        moore.constraint.expr %gt : i1
      }
    }
  }
}

// CHECK-LABEL: func.func @test_foreach_implication
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_foreach_implication(%obj: !moore.class<@ForeachImplication>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: moore.constraint.foreach
  // CHECK-NOT: moore.constraint.implication
  %success = moore.randomize %obj : !moore.class<@ForeachImplication>
  return %success : i1
}

/// Test foreach with array element as implication condition.
/// This pattern: foreach (arr[i]) arr[i] -> constraint;
/// Uses the array element value as the antecedent of the implication.
///
/// Corresponds to SystemVerilog:
///   rand bit flags[4];
///   rand int values[4];
///   constraint c { foreach (flags[i]) { flags[i] -> values[i] > 0; } }

moore.class.classdecl @ForeachElementImplication {
  moore.class.propertydecl @flags : !moore.uarray<4 x i1> rand_mode rand
  moore.class.propertydecl @values : !moore.uarray<4 x i32> rand_mode rand
  moore.constraint.block @c_elem_impl {
  ^bb0(%flags: !moore.uarray<4 x i1>, %values: !moore.uarray<4 x i32>):
    moore.constraint.foreach %flags : !moore.uarray<4 x i1> {
    ^bb0(%i: !moore.i32):
      // Get the flag element - use it as implication antecedent
      %flag_elem = moore.dyn_extract %flags from %i : !moore.uarray<4 x i1>, !moore.i32 -> !moore.i1
      moore.constraint.implication %flag_elem : i1 {
        // If flag[i] is true, then value[i] must be > 0
        %val_elem = moore.dyn_extract %values from %i : !moore.uarray<4 x i32>, !moore.i32 -> !moore.i32
        %c0 = moore.constant 0 : i32
        %gt = moore.sgt %val_elem, %c0 : i32 -> i1
        moore.constraint.expr %gt : i1
      }
    }
  }
}

// CHECK-LABEL: func.func @test_foreach_element_implication
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_foreach_element_implication(%obj: !moore.class<@ForeachElementImplication>) -> i1 {
  // The foreach and implication constraints should both be erased
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: moore.constraint.foreach
  // CHECK-NOT: moore.constraint.implication
  %success = moore.randomize %obj : !moore.class<@ForeachElementImplication>
  return %success : i1
}

/// Test foreach with if-else constraint inside.
/// This pattern: foreach (arr[i]) if (cond) constraint;
///
/// Corresponds to SystemVerilog:
///   rand bit [7:0] data[8];
///   rand bit mode;
///   constraint c {
///     foreach (data[i]) {
///       if (mode) data[i] inside {[0:127]};
///       else data[i] inside {[128:255]};
///     }
///   }

moore.class.classdecl @ForeachIfElse {
  moore.class.propertydecl @mode : !moore.i1 rand_mode rand
  moore.class.propertydecl @data : !moore.uarray<8 x i8> rand_mode rand
  moore.constraint.block @c_foreach_ifelse {
  ^bb0(%mode: !moore.i1, %data: !moore.uarray<8 x i8>):
    moore.constraint.foreach %data : !moore.uarray<8 x i8> {
    ^bb0(%i: !moore.i32):
      %elem = moore.dyn_extract %data from %i : !moore.uarray<8 x i8>, !moore.i32 -> !moore.i8
      moore.constraint.if_else %mode : i1 {
        // If mode is true: data[i] in [0, 127]
        %c0 = moore.constant 0 : i8
        %c127 = moore.constant 127 : i8
        %ge = moore.uge %elem, %c0 : i8 -> i1
        %le = moore.ule %elem, %c127 : i8 -> i1
        %in_low = moore.and %ge, %le : !moore.i1
        moore.constraint.expr %in_low : i1
      } else {
        // Else: data[i] in [128, 255]
        %c128 = moore.constant 128 : i8
        %c255 = moore.constant 255 : i8
        %ge = moore.uge %elem, %c128 : i8 -> i1
        %le = moore.ule %elem, %c255 : i8 -> i1
        %in_high = moore.and %ge, %le : !moore.i1
        moore.constraint.expr %in_high : i1
      }
    }
  }
}

// CHECK-LABEL: func.func @test_foreach_ifelse
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_foreach_ifelse(%obj: !moore.class<@ForeachIfElse>) -> i1 {
  // Both foreach and if-else should be erased
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: moore.constraint.foreach
  // CHECK-NOT: moore.constraint.if_else
  %success = moore.randomize %obj : !moore.class<@ForeachIfElse>
  return %success : i1
}

/// Test foreach with if-only constraint (no else branch).
/// This pattern: foreach (arr[i]) if (cond) constraint;
///
/// Corresponds to SystemVerilog:
///   rand int arr[4];
///   rand bit valid;
///   constraint c { foreach (arr[i]) { if (valid) arr[i] > 0; } }

moore.class.classdecl @ForeachIfOnly {
  moore.class.propertydecl @valid : !moore.i1 rand_mode rand
  moore.class.propertydecl @arr : !moore.uarray<4 x i32> rand_mode rand
  moore.constraint.block @c_foreach_if_only {
  ^bb0(%valid: !moore.i1, %arr: !moore.uarray<4 x i32>):
    moore.constraint.foreach %arr : !moore.uarray<4 x i32> {
    ^bb0(%i: !moore.i32):
      moore.constraint.if_else %valid : i1 {
        %elem = moore.dyn_extract %arr from %i : !moore.uarray<4 x i32>, !moore.i32 -> !moore.i32
        %c0 = moore.constant 0 : i32
        %gt = moore.sgt %elem, %c0 : i32 -> i1
        moore.constraint.expr %gt : i1
      }
    }
  }
}

// CHECK-LABEL: func.func @test_foreach_if_only
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_foreach_if_only(%obj: !moore.class<@ForeachIfOnly>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: moore.constraint.foreach
  // CHECK-NOT: moore.constraint.if_else
  %success = moore.randomize %obj : !moore.class<@ForeachIfOnly>
  return %success : i1
}

/// Test nested foreach with implication.
/// This pattern: foreach (a[i]) foreach (b[j]) (i == j) -> constraint;
///
/// Corresponds to SystemVerilog:
///   rand int a[4];
///   rand int b[4];
///   constraint c {
///     foreach (a[i]) {
///       foreach (b[j]) {
///         (i == j) -> (a[i] == b[j]);
///       }
///     }
///   }

moore.class.classdecl @NestedForeachImplication {
  moore.class.propertydecl @a : !moore.uarray<4 x i32> rand_mode rand
  moore.class.propertydecl @b : !moore.uarray<4 x i32> rand_mode rand
  moore.constraint.block @c_nested {
  ^bb0(%a: !moore.uarray<4 x i32>, %b: !moore.uarray<4 x i32>):
    moore.constraint.foreach %a : !moore.uarray<4 x i32> {
    ^bb0(%i: !moore.i32):
      moore.constraint.foreach %b : !moore.uarray<4 x i32> {
      ^bb0(%j: !moore.i32):
        %idx_eq = moore.eq %i, %j : i32 -> i1
        moore.constraint.implication %idx_eq : i1 {
          %a_elem = moore.dyn_extract %a from %i : !moore.uarray<4 x i32>, !moore.i32 -> !moore.i32
          %b_elem = moore.dyn_extract %b from %j : !moore.uarray<4 x i32>, !moore.i32 -> !moore.i32
          %elem_eq = moore.eq %a_elem, %b_elem : i32 -> i1
          moore.constraint.expr %elem_eq : i1
        }
      }
    }
  }
}

// CHECK-LABEL: func.func @test_nested_foreach_implication
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_nested_foreach_implication(%obj: !moore.class<@NestedForeachImplication>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: moore.constraint.foreach
  // CHECK-NOT: moore.constraint.implication
  %success = moore.randomize %obj : !moore.class<@NestedForeachImplication>
  return %success : i1
}

/// Test foreach with combined index-based implication.
/// This pattern uses the loop index in the implication condition.
///
/// Corresponds to SystemVerilog:
///   rand int data[8];
///   constraint c {
///     foreach (data[i]) {
///       (i < 4) -> (data[i] < 100);
///       (i >= 4) -> (data[i] >= 100);
///     }
///   }

moore.class.classdecl @ForeachIndexImplication {
  moore.class.propertydecl @data : !moore.uarray<8 x i32> rand_mode rand
  moore.constraint.block @c_idx_impl {
  ^bb0(%data: !moore.uarray<8 x i32>):
    moore.constraint.foreach %data : !moore.uarray<8 x i32> {
    ^bb0(%i: !moore.i32):
      %c4 = moore.constant 4 : i32
      %c100 = moore.constant 100 : i32
      %i_lt_4 = moore.slt %i, %c4 : i32 -> i1
      %i_ge_4 = moore.sge %i, %c4 : i32 -> i1
      %elem = moore.dyn_extract %data from %i : !moore.uarray<8 x i32>, !moore.i32 -> !moore.i32
      // First implication: i < 4 -> data[i] < 100
      moore.constraint.implication %i_lt_4 : i1 {
        %elem_lt = moore.slt %elem, %c100 : i32 -> i1
        moore.constraint.expr %elem_lt : i1
      }
      // Second implication: i >= 4 -> data[i] >= 100
      moore.constraint.implication %i_ge_4 : i1 {
        %elem_ge = moore.sge %elem, %c100 : i32 -> i1
        moore.constraint.expr %elem_ge : i1
      }
    }
  }
}

// CHECK-LABEL: func.func @test_foreach_index_implication
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_foreach_index_implication(%obj: !moore.class<@ForeachIndexImplication>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: moore.constraint.foreach
  // CHECK-NOT: moore.constraint.implication
  %success = moore.randomize %obj : !moore.class<@ForeachIndexImplication>
  return %success : i1
}

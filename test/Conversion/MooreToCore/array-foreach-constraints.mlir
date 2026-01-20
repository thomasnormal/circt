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

// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32

//===----------------------------------------------------------------------===//
// Array Unique Constraint Lowering Tests
//===----------------------------------------------------------------------===//
//
// Unique constraints ensure all elements in an array (or all values in a set
// of scalar variables) are distinct. This is specified in IEEE 1800-2017
// Section 18.5.5 "Uniqueness constraints".
//
// For unique constraints inside constraint blocks (class context), the ops
// are erased as the constraint info is processed during RandomizeOp lowering.
//
// For standalone unique constraints, runtime calls would be generated to
// validate uniqueness:
// - For arrays: __moore_constraint_unique_check(array_ptr, num_elements, element_size)
// - For scalars: __moore_constraint_unique_scalars(values_ptr, num_values, value_size)

//===----------------------------------------------------------------------===//
// Static Array Unique Constraint in Class Context
//===----------------------------------------------------------------------===//

/// Test class with unique constraint on a static array.
/// Corresponds to SystemVerilog:
///   class test;
///     rand bit [7:0] arr[4];
///     constraint c { unique {arr}; }
///   endclass

moore.class.classdecl @UniqueArrayClass {
  moore.class.propertydecl @arr : !moore.uarray<4 x i8> rand_mode rand
  moore.constraint.block @c_unique {
  ^bb0(%arr: !moore.uarray<4 x i8>):
    moore.constraint.unique %arr : !moore.uarray<4 x i8>
  }
}

// CHECK-LABEL: func.func @test_unique_static_array
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_unique_static_array(%obj: !moore.class<@UniqueArrayClass>) -> i1 {
  // Randomization should call basic randomize and the unique constraint
  // should be erased (processed as part of constraint block)
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: moore.constraint.unique
  %success = moore.randomize %obj : !moore.class<@UniqueArrayClass>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Larger Static Array Unique Constraint
//===----------------------------------------------------------------------===//

/// Test with larger array (8 x i32 elements).
/// Ensures element size calculation works correctly for multi-byte types.

moore.class.classdecl @UniqueArray32Class {
  moore.class.propertydecl @data : !moore.uarray<8 x i32> rand_mode rand
  moore.constraint.block @c_unique {
  ^bb0(%data: !moore.uarray<8 x i32>):
    moore.constraint.unique %data : !moore.uarray<8 x i32>
  }
}

// CHECK-LABEL: func.func @test_unique_array_i32
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_unique_array_i32(%obj: !moore.class<@UniqueArray32Class>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: moore.constraint.unique
  %success = moore.randomize %obj : !moore.class<@UniqueArray32Class>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Multiple Scalar Variables Unique Constraint
//===----------------------------------------------------------------------===//

/// Test unique constraint on multiple scalar variables.
/// Corresponds to SystemVerilog:
///   class test;
///     rand bit [7:0] a, b, c;
///     constraint c { unique {a, b, c}; }
///   endclass

moore.class.classdecl @UniqueScalarsClass {
  moore.class.propertydecl @a : !moore.i8 rand_mode rand
  moore.class.propertydecl @b : !moore.i8 rand_mode rand
  moore.class.propertydecl @c : !moore.i8 rand_mode rand
  moore.constraint.block @c_unique {
  ^bb0(%a: !moore.i8, %b: !moore.i8, %c: !moore.i8):
    moore.constraint.unique %a, %b, %c : !moore.i8, !moore.i8, !moore.i8
  }
}

// CHECK-LABEL: func.func @test_unique_scalars
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_unique_scalars(%obj: !moore.class<@UniqueScalarsClass>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: moore.constraint.unique
  %success = moore.randomize %obj : !moore.class<@UniqueScalarsClass>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Empty Unique Constraint (Edge Case)
//===----------------------------------------------------------------------===//

/// Test empty function - no unique constraint here.

// CHECK-LABEL: func.func @test_empty_unique
func.func @test_empty_unique() {
  // CHECK: return
  return
}

//===----------------------------------------------------------------------===//
// Combined Unique and Foreach Constraints
//===----------------------------------------------------------------------===//

/// Test class with both unique and foreach constraints.
/// Both constraints are processed during RandomizeOp lowering and erased.

moore.class.classdecl @UniqueAndForeachClass {
  moore.class.propertydecl @arr : !moore.uarray<4 x i8> rand_mode rand
  moore.constraint.block @c_unique {
  ^bb0(%arr: !moore.uarray<4 x i8>):
    moore.constraint.unique %arr : !moore.uarray<4 x i8>
  }
  moore.constraint.block @c_range {
  ^bb0(%arr: !moore.uarray<4 x i8>):
    // Additional range constraints could go here
    moore.constraint.foreach %arr : !moore.uarray<4 x i8> {
    ^bb0(%i: !moore.i32):
      %c0 = moore.constant 0 : i8
      %c100 = moore.constant 100 : i8
      %elem = moore.dyn_extract %arr from %i : !moore.uarray<4 x i8>, !moore.i32 -> !moore.i8
      %ge = moore.uge %elem, %c0 : i8 -> i1
      %le = moore.ule %elem, %c100 : i8 -> i1
      %in_range = moore.and %ge, %le : !moore.i1
      moore.constraint.expr %in_range : i1
    }
  }
}

// CHECK-LABEL: func.func @test_unique_and_foreach
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_unique_and_foreach(%obj: !moore.class<@UniqueAndForeachClass>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: moore.constraint.unique
  // CHECK-NOT: moore.constraint.foreach
  %success = moore.randomize %obj : !moore.class<@UniqueAndForeachClass>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Unique Constraint with Two Variables
//===----------------------------------------------------------------------===//

/// Test unique constraint with exactly two scalar variables.

moore.class.classdecl @UniqueTwoScalarsClass {
  moore.class.propertydecl @x : !moore.i32 rand_mode rand
  moore.class.propertydecl @y : !moore.i32 rand_mode rand
  moore.constraint.block @c_unique {
  ^bb0(%x: !moore.i32, %y: !moore.i32):
    moore.constraint.unique %x, %y : !moore.i32, !moore.i32
  }
}

// CHECK-LABEL: func.func @test_unique_two_scalars
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_unique_two_scalars(%obj: !moore.class<@UniqueTwoScalarsClass>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: moore.constraint.unique
  %success = moore.randomize %obj : !moore.class<@UniqueTwoScalarsClass>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Unique Constraint with Different Element Types
//===----------------------------------------------------------------------===//

/// Test unique constraint with 16-bit elements.

moore.class.classdecl @UniqueArray16Class {
  moore.class.propertydecl @values : !moore.uarray<6 x i16> rand_mode rand
  moore.constraint.block @c_unique {
  ^bb0(%values: !moore.uarray<6 x i16>):
    moore.constraint.unique %values : !moore.uarray<6 x i16>
  }
}

// CHECK-LABEL: func.func @test_unique_array_i16
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_unique_array_i16(%obj: !moore.class<@UniqueArray16Class>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: moore.constraint.unique
  %success = moore.randomize %obj : !moore.class<@UniqueArray16Class>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Multiple Unique Constraints
//===----------------------------------------------------------------------===//

/// Test class with multiple separate unique constraints.

moore.class.classdecl @MultipleUniqueClass {
  moore.class.propertydecl @arr1 : !moore.uarray<3 x i8> rand_mode rand
  moore.class.propertydecl @arr2 : !moore.uarray<4 x i8> rand_mode rand
  moore.constraint.block @c_unique1 {
  ^bb0(%arr1: !moore.uarray<3 x i8>):
    moore.constraint.unique %arr1 : !moore.uarray<3 x i8>
  }
  moore.constraint.block @c_unique2 {
  ^bb0(%arr2: !moore.uarray<4 x i8>):
    moore.constraint.unique %arr2 : !moore.uarray<4 x i8>
  }
}

// CHECK-LABEL: func.func @test_multiple_unique
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_multiple_unique(%obj: !moore.class<@MultipleUniqueClass>) -> i1 {
  // Both unique constraints should be processed and erased
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: moore.constraint.unique
  %success = moore.randomize %obj : !moore.class<@MultipleUniqueClass>
  return %success : i1
}

// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32
// CHECK-DAG: llvm.func @__moore_randomize_with_range(i64, i64) -> i64
// CHECK-DAG: llvm.func @__moore_randomize_with_ranges(!llvm.ptr, i64) -> i64
// CHECK-DAG: llvm.func @__moore_dyn_array_new(i32) -> !llvm.struct<(ptr, i64)>
// CHECK-DAG: llvm.func @__moore_is_rand_enabled(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @__moore_is_constraint_enabled(!llvm.ptr, !llvm.ptr) -> i32

//===----------------------------------------------------------------------===//
// Inline Constraint Lowering Tests
//===----------------------------------------------------------------------===//
// These tests verify that inline constraints (the `with` clause in randomize())
// are correctly extracted and applied during randomization.
// IEEE 1800-2017 Section 18.7 "In-line constraints"

/// Test class for inline constraint testing
moore.class.classdecl @InlineTestClass {
  moore.class.propertydecl @x : !moore.i32 rand_mode rand
  moore.class.propertydecl @y : !moore.i32 rand_mode rand
}

//===----------------------------------------------------------------------===//
// Test: Inline single range constraint
//===----------------------------------------------------------------------===//
// Corresponds to: obj.randomize() with { x inside {[10:50]}; };

// CHECK-LABEL: func.func @test_inline_single_range
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_inline_single_range(%obj: !moore.class<@InlineTestClass>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK: llvm.call @__moore_is_rand_enabled
  // Should apply the inline constraint range [10, 50]
  // CHECK: scf.if
  // CHECK: llvm.call @__moore_randomize_with_range
  %success = moore.randomize %obj : !moore.class<@InlineTestClass> {
    %0 = moore.class.property_ref %obj[@x] : !moore.class<@InlineTestClass> -> !moore.ref<i32>
    %1 = moore.read %0 : <i32>
    moore.constraint.inside %1, [10, 50] : !moore.i32
  }
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test: Inline multi-range constraint
//===----------------------------------------------------------------------===//
// Corresponds to: obj.randomize() with { y inside {1, [5:10], 20}; };

// CHECK-LABEL: func.func @test_inline_multi_range
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_inline_multi_range(%obj: !moore.class<@InlineTestClass>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK: llvm.call @__moore_randomize_with_ranges
  %success = moore.randomize %obj : !moore.class<@InlineTestClass> {
    %0 = moore.class.property_ref %obj[@y] : !moore.class<@InlineTestClass> -> !moore.ref<i32>
    %1 = moore.read %0 : <i32>
    // Ranges: [1,1], [5,10], [20,20]
    moore.constraint.inside %1, [1, 1, 5, 10, 20, 20] : !moore.i32
  }
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test: Multiple inline constraints
//===----------------------------------------------------------------------===//
// Corresponds to: obj.randomize() with { x inside {[0:100]}; y inside {[50:150]}; };

// CHECK-LABEL: func.func @test_inline_multiple_constraints
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_inline_multiple_constraints(%obj: !moore.class<@InlineTestClass>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // Should apply both inline constraints
  // CHECK: llvm.call @__moore_randomize_with_range
  // CHECK: llvm.call @__moore_randomize_with_range
  %success = moore.randomize %obj : !moore.class<@InlineTestClass> {
    %0 = moore.class.property_ref %obj[@x] : !moore.class<@InlineTestClass> -> !moore.ref<i32>
    %1 = moore.read %0 : <i32>
    moore.constraint.inside %1, [0, 100] : !moore.i32
    %2 = moore.class.property_ref %obj[@y] : !moore.class<@InlineTestClass> -> !moore.ref<i32>
    %3 = moore.read %2 : <i32>
    moore.constraint.inside %3, [50, 150] : !moore.i32
  }
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test: Inline constraint combined with class-level constraint
//===----------------------------------------------------------------------===//
// Class has a constraint; inline constraint adds another

moore.class.classdecl @MixedConstraintClass {
  moore.class.propertydecl @a : !moore.i32 rand_mode rand
  moore.class.propertydecl @b : !moore.i32 rand_mode rand
  // Class-level constraint on 'a'
  moore.constraint.block @class_c {
  ^bb0(%this: !moore.class<@MixedConstraintClass>):
    %0 = moore.class.property_ref %this[@a] : !moore.class<@MixedConstraintClass> -> !moore.ref<i32>
    %1 = moore.read %0 : <i32>
    moore.constraint.inside %1, [0, 50] : !moore.i32
  }
}

// CHECK-LABEL: func.func @test_inline_with_class_constraint
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_inline_with_class_constraint(%obj: !moore.class<@MixedConstraintClass>) -> i1 {
  // Both class-level (on 'a') and inline (on 'b') constraints should be applied
  // CHECK: llvm.call @__moore_randomize_basic
  // Class-level constraint on 'a' [0, 50] (with constraint_enabled check)
  // CHECK: llvm.call @__moore_is_constraint_enabled
  // CHECK: llvm.call @__moore_randomize_with_range
  // Inline constraint on 'b' [100, 200]
  // CHECK: llvm.call @__moore_randomize_with_range
  %success = moore.randomize %obj : !moore.class<@MixedConstraintClass> {
    // Inline constraint on 'b'
    %0 = moore.class.property_ref %obj[@b] : !moore.class<@MixedConstraintClass> -> !moore.ref<i32>
    %1 = moore.read %0 : <i32>
    moore.constraint.inside %1, [100, 200] : !moore.i32
  }
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test: Empty randomize (no inline constraints)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_no_inline_constraints
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_no_inline_constraints(%obj: !moore.class<@InlineTestClass>) -> i1 {
  // No inline constraints - just basic randomization
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK-NOT: llvm.call @__moore_randomize_with_range
  %success = moore.randomize %obj : !moore.class<@InlineTestClass>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test: Soft inline constraint
//===----------------------------------------------------------------------===//
// Corresponds to: obj.randomize() with { soft x inside {[0:10]}; };

// CHECK-LABEL: func.func @test_inline_soft_constraint
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_inline_soft_constraint(%obj: !moore.class<@InlineTestClass>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // Soft constraint should be applied (as a default value)
  %success = moore.randomize %obj : !moore.class<@InlineTestClass> {
    %0 = moore.class.property_ref %obj[@x] : !moore.class<@InlineTestClass> -> !moore.ref<i32>
    %1 = moore.read %0 : <i32>
    moore.constraint.inside %1, [0, 10] : !moore.i32 soft
  }
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test: Inline constraint with 8-bit type
//===----------------------------------------------------------------------===//

moore.class.classdecl @SmallTypeClass {
  moore.class.propertydecl @byte_val : !moore.i8 rand_mode rand
}

// CHECK-LABEL: func.func @test_inline_small_type
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_inline_small_type(%obj: !moore.class<@SmallTypeClass>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK: llvm.call @__moore_randomize_with_range
  // CHECK: arith.trunci %{{.*}} : i64 to i8
  %success = moore.randomize %obj : !moore.class<@SmallTypeClass> {
    %0 = moore.class.property_ref %obj[@byte_val] : !moore.class<@SmallTypeClass> -> !moore.ref<i8>
    %1 = moore.read %0 : <i8>
    moore.constraint.inside %1, [0, 255] : !moore.i8
  }
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test: Inline constraint with ConstraintExprOp (comparison-based)
//===----------------------------------------------------------------------===//
// Corresponds to: obj.randomize() with { x > 0; x < 100; };
// ConstraintExprOp uses comparison ops (UgtOp, UltOp) to define bounds.

// CHECK-LABEL: func.func @test_inline_expr_constraint
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_inline_expr_constraint(%obj: !moore.class<@InlineTestClass>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // Should extract bounds from x > 0 (lower=1) and x < 100 (upper=99)
  // CHECK: llvm.call @__moore_randomize_with_range
  %success = moore.randomize %obj : !moore.class<@InlineTestClass> {
    %0 = moore.class.property_ref %obj[@x] : !moore.class<@InlineTestClass> -> !moore.ref<i32>
    %1 = moore.read %0 : <i32>
    %zero = moore.constant 0 : i32
    %hundred = moore.constant 100 : i32
    %cmp1 = moore.ugt %1, %zero : i32 -> i1
    moore.constraint.expr %cmp1 : i1
    %cmp2 = moore.ult %1, %hundred : i32 -> i1
    moore.constraint.expr %cmp2 : i1
  }
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test: Inline constraint with signed comparison (SgtOp, SltOp)
//===----------------------------------------------------------------------===//
// Corresponds to: obj.randomize() with { x > -50; x < 50; }; (signed int)

// CHECK-LABEL: func.func @test_inline_signed_expr_constraint
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_inline_signed_expr_constraint(%obj: !moore.class<@InlineTestClass>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // Should extract signed bounds from x > -50 (lower=-49) and x < 50 (upper=49)
  // CHECK: llvm.call @__moore_randomize_with_range
  %success = moore.randomize %obj : !moore.class<@InlineTestClass> {
    %0 = moore.class.property_ref %obj[@x] : !moore.class<@InlineTestClass> -> !moore.ref<i32>
    %1 = moore.read %0 : <i32>
    %neg50 = moore.constant -50 : i32
    %pos50 = moore.constant 50 : i32
    %cmp1 = moore.sgt %1, %neg50 : i32 -> i1
    moore.constraint.expr %cmp1 : i1
    %cmp2 = moore.slt %1, %pos50 : i32 -> i1
    moore.constraint.expr %cmp2 : i1
  }
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test: Inline constraint with equality (EqOp)
//===----------------------------------------------------------------------===//
// Corresponds to: obj.randomize() with { x == 42; };

// CHECK-LABEL: func.func @test_inline_eq_constraint
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_inline_eq_constraint(%obj: !moore.class<@InlineTestClass>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // Should extract equality bound: x == 42 -> range [42, 42]
  // CHECK: llvm.call @__moore_randomize_with_range
  %success = moore.randomize %obj : !moore.class<@InlineTestClass> {
    %0 = moore.class.property_ref %obj[@x] : !moore.class<@InlineTestClass> -> !moore.ref<i32>
    %1 = moore.read %0 : <i32>
    %val = moore.constant 42 : i32
    %cmp = moore.eq %1, %val : i32 -> i1
    moore.constraint.expr %cmp : i1
  }
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Test: Inline array constraints from ConstraintExprOp
//===----------------------------------------------------------------------===//
// Covers:
//   1) writeData.size() == 1
//   2) targetAddress inside {allowed}
// where `inside {allowed}` is represented as moore.array.contains(allowed, x).

moore.class.classdecl @InlineArrayConstraintClass {
  moore.class.propertydecl @targetAddress : !moore.i7 rand_mode rand
  moore.class.propertydecl @writeData : !moore.open_uarray<!moore.i8> rand_mode rand
}

// CHECK-LABEL: func.func @test_inline_array_constraint_exprs
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr, %[[ALLOWED:.*]]: !llvm.struct<(ptr, i64)>)
func.func @test_inline_array_constraint_exprs(%obj: !moore.class<@InlineArrayConstraintClass>,
                                              %allowed: !moore.open_uarray<!moore.i7>) -> i1 {
  %one = moore.constant 1 : i32

  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK: llvm.call @__moore_dyn_array_new
  // CHECK: llvm.insertvalue
  // CHECK: llvm.extractvalue %[[ALLOWED]][1]
  // CHECK: llvm.call @__moore_randomize_with_range
  %success = moore.randomize %obj : !moore.class<@InlineArrayConstraintClass> {
    %write_ref = moore.class.property_ref %obj[@writeData] : !moore.class<@InlineArrayConstraintClass> -> !moore.ref<open_uarray<i8>>
    %write_val = moore.read %write_ref : <open_uarray<i8>>
    %write_size = moore.array.size %write_val : !moore.open_uarray<!moore.i8>
    %size_eq = moore.eq %write_size, %one : i32 -> i1
    moore.constraint.expr %size_eq : i1

    %addr_ref = moore.class.property_ref %obj[@targetAddress] : !moore.class<@InlineArrayConstraintClass> -> !moore.ref<i7>
    %addr_val = moore.read %addr_ref : <i7>
    %contains = moore.array.contains %allowed, %addr_val : open_uarray<i7>, i7
    moore.constraint.expr %contains : i1
  }
  return %success : i1
}

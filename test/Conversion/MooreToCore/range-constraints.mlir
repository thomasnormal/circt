// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32
// CHECK-DAG: llvm.func @__moore_randomize_with_range(i64, i64) -> i64

//===----------------------------------------------------------------------===//
// Range Constraint Support Tests
//===----------------------------------------------------------------------===//

/// Test class with simple range constraint using ConstraintInsideOp
/// Corresponds to SystemVerilog: constraint range_c { value inside {[1:99]}; }

moore.class.classdecl @RangeConstrainedClass {
  moore.class.propertydecl @value : !moore.i32 rand_mode rand
  moore.constraint.block @range_c {
  ^bb0(%value: !moore.i32):
    // Constraint: value inside {[1:99]} - single range [1, 99]
    moore.constraint.inside %value, [1, 99] : !moore.i32
  }
}

// CHECK-LABEL: func.func @test_range_constraint
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_range_constraint(%obj: !moore.class<@RangeConstrainedClass>) -> i1 {
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(8 : i64) : i64
  // CHECK: llvm.call @__moore_randomize_basic(%[[OBJ]], %[[SIZE]]) : (!llvm.ptr, i64) -> i32
  // CHECK: %[[MIN:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: %[[MAX:.*]] = llvm.mlir.constant(99 : i64) : i64
  // CHECK: %[[RANGE_RESULT:.*]] = llvm.call @__moore_randomize_with_range(%[[MIN]], %[[MAX]]) : (i64, i64) -> i64
  // CHECK: %[[TRUNC:.*]] = arith.trunci %[[RANGE_RESULT]] : i64 to i32
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[OBJ]][0, 1]
  // CHECK: llvm.store %[[TRUNC]], %[[GEP]] : i32, !llvm.ptr
  // CHECK: %[[SUCCESS:.*]] = hw.constant true
  // CHECK: return %[[SUCCESS]] : i1
  %success = moore.randomize %obj : !moore.class<@RangeConstrainedClass>
  return %success : i1
}

/// Test class without constraints - should use basic randomization only

moore.class.classdecl @UnconstrainedClass {
  moore.class.propertydecl @data : !moore.i32 rand_mode rand
}

// CHECK-LABEL: func.func @test_unconstrained
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_unconstrained(%obj: !moore.class<@UnconstrainedClass>) -> i1 {
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(8 : i64) : i64
  // CHECK: %[[RESULT:.*]] = llvm.call @__moore_randomize_basic(%[[OBJ]], %[[SIZE]]) : (!llvm.ptr, i64) -> i32
  // CHECK: %[[SUCCESS:.*]] = arith.trunci %[[RESULT]] : i32 to i1
  // CHECK: return %[[SUCCESS]] : i1
  %success = moore.randomize %obj : !moore.class<@UnconstrainedClass>
  return %success : i1
}

/// Test class with constraint block but empty body - should use basic randomization

moore.class.classdecl @EmptyConstraintClass {
  moore.class.propertydecl @x : !moore.i32 rand_mode rand
  moore.constraint.block @empty_constraint {
  }
}

// CHECK-LABEL: func.func @test_empty_constraint
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_empty_constraint(%obj: !moore.class<@EmptyConstraintClass>) -> i1 {
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(8 : i64) : i64
  // CHECK: %[[RESULT:.*]] = llvm.call @__moore_randomize_basic(%[[OBJ]], %[[SIZE]]) : (!llvm.ptr, i64) -> i32
  // CHECK: %[[SUCCESS:.*]] = arith.trunci %[[RESULT]] : i32 to i1
  // CHECK: return %[[SUCCESS]] : i1
  %success = moore.randomize %obj : !moore.class<@EmptyConstraintClass>
  return %success : i1
}

/// Test class with multiple properties but only one constrained

moore.class.classdecl @PartialConstraintClass {
  moore.class.propertydecl @constrained : !moore.i32 rand_mode rand
  moore.class.propertydecl @unconstrained : !moore.i32 rand_mode rand
  moore.constraint.block @partial_c {
  ^bb0(%constrained: !moore.i32):
    // Only constrained property is in the constraint
    moore.constraint.inside %constrained, [10, 20] : !moore.i32
  }
}

// CHECK-LABEL: func.func @test_partial_constraint
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_partial_constraint(%obj: !moore.class<@PartialConstraintClass>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK: %[[MIN:.*]] = llvm.mlir.constant(10 : i64) : i64
  // CHECK: %[[MAX:.*]] = llvm.mlir.constant(20 : i64) : i64
  // CHECK: llvm.call @__moore_randomize_with_range(%[[MIN]], %[[MAX]])
  %success = moore.randomize %obj : !moore.class<@PartialConstraintClass>
  return %success : i1
}

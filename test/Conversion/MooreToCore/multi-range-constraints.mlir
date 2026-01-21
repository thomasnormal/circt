// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32
// CHECK-DAG: llvm.func @__moore_randomize_with_ranges(!llvm.ptr, i64) -> i64
// CHECK-DAG: llvm.func @__moore_randomize_with_range(i64, i64) -> i64
// CHECK-DAG: llvm.func @__moore_is_rand_enabled(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @__moore_is_constraint_enabled(!llvm.ptr, !llvm.ptr) -> i32

//===----------------------------------------------------------------------===//
// Multi-Range Constraint Support Tests
//===----------------------------------------------------------------------===//

/// Test class with multi-range inside constraint
/// Corresponds to SystemVerilog: constraint multi_c { value inside {[1:10], [20:30], [50:60]}; }

moore.class.classdecl @MultiRangeClass {
  moore.class.propertydecl @value : !moore.i32 rand_mode rand
  moore.constraint.block @multi_c {
  ^bb0(%value: !moore.i32):
    // Constraint: value inside {[1:10], [20:30], [50:60]}
    // Represented as array of pairs: [1, 10, 20, 30, 50, 60]
    moore.constraint.inside %value, [1, 10, 20, 30, 50, 60] : !moore.i32
  }
}

// CHECK-LABEL: func.func @test_multi_range_constraint
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_multi_range_constraint(%obj: !moore.class<@MultiRangeClass>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic(%[[OBJ]], {{.*}}) : (!llvm.ptr, i64) -> i32
  // CHECK: llvm.call @__moore_is_rand_enabled
  // CHECK: llvm.call @__moore_is_constraint_enabled
  // CHECK: scf.if
  // Multi-range: allocate array, store range pairs, call __moore_randomize_with_ranges
  // CHECK: llvm.alloca {{.*}} x !llvm.array<6 x i64> : (i64) -> !llvm.ptr
  // Store range pairs and call the function
  // CHECK: llvm.call @__moore_randomize_with_ranges({{.*}}, {{.*}}) : (!llvm.ptr, i64) -> i64
  // Truncate to i32 and store
  // CHECK: arith.trunci {{.*}} : i64 to i32
  // CHECK: llvm.store {{.*}} : i32, !llvm.ptr
  // CHECK: return %{{.*}} : i1
  %success = moore.randomize %obj : !moore.class<@MultiRangeClass>
  return %success : i1
}

/// Test class with two-range inside constraint (edge case between single and multi)
/// Corresponds to SystemVerilog: constraint two_c { value inside {[0:50], [100:150]}; }

moore.class.classdecl @TwoRangeClass {
  moore.class.propertydecl @value : !moore.i32 rand_mode rand
  moore.constraint.block @two_c {
  ^bb0(%value: !moore.i32):
    // Constraint: value inside {[0:50], [100:150]}
    moore.constraint.inside %value, [0, 50, 100, 150] : !moore.i32
  }
}

// CHECK-LABEL: func.func @test_two_range_constraint
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_two_range_constraint(%obj: !moore.class<@TwoRangeClass>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK: llvm.alloca {{.*}} x !llvm.array<4 x i64>
  // CHECK: llvm.call @__moore_randomize_with_ranges({{.*}}, {{.*}}) : (!llvm.ptr, i64) -> i64
  %success = moore.randomize %obj : !moore.class<@TwoRangeClass>
  return %success : i1
}

/// Test that single-range constraints still use __moore_randomize_with_range
/// (backward compatibility check)

moore.class.classdecl @SingleRangeClass {
  moore.class.propertydecl @value : !moore.i32 rand_mode rand
  moore.constraint.block @single_c {
  ^bb0(%value: !moore.i32):
    // Single range: value inside {[10:90]}
    moore.constraint.inside %value, [10, 90] : !moore.i32
  }
}

// CHECK-LABEL: func.func @test_single_range_still_works
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_single_range_still_works(%obj: !moore.class<@SingleRangeClass>) -> i1 {
  // CHECK-DAG: %[[MIN:.*]] = llvm.mlir.constant(10 : i64) : i64
  // CHECK-DAG: %[[MAX:.*]] = llvm.mlir.constant(90 : i64) : i64
  // CHECK: llvm.call @__moore_randomize_basic
  // Single range should use __moore_randomize_with_range, NOT __moore_randomize_with_ranges
  // CHECK: llvm.call @__moore_randomize_with_range(%[[MIN]], %[[MAX]]) : (i64, i64) -> i64
  // CHECK-NOT: llvm.call @__moore_randomize_with_ranges
  %success = moore.randomize %obj : !moore.class<@SingleRangeClass>
  return %success : i1
}

/// Test class with multiple properties, one with multi-range constraint

moore.class.classdecl @MixedConstraintClass {
  moore.class.propertydecl @multi : !moore.i32 rand_mode rand
  moore.class.propertydecl @single : !moore.i32 rand_mode rand
  moore.constraint.block @constraints {
  ^bb0(%multi: !moore.i32, %single: !moore.i32):
    // Multi-range constraint on first property
    moore.constraint.inside %multi, [1, 5, 10, 15, 20, 25] : !moore.i32
    // Single-range constraint on second property
    moore.constraint.inside %single, [100, 200] : !moore.i32
  }
}

// CHECK-LABEL: func.func @test_mixed_constraints
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_mixed_constraints(%obj: !moore.class<@MixedConstraintClass>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // Multi-range constraint uses __moore_randomize_with_ranges
  // CHECK: llvm.alloca {{.*}} x !llvm.array<6 x i64>
  // CHECK: llvm.call @__moore_randomize_with_ranges
  // Single-range constraint uses __moore_randomize_with_range
  // CHECK: llvm.call @__moore_randomize_with_range
  %success = moore.randomize %obj : !moore.class<@MixedConstraintClass>
  return %success : i1
}

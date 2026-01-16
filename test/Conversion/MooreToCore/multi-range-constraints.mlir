// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32
// CHECK-DAG: llvm.func @__moore_randomize_with_ranges(!llvm.ptr, i64) -> i64

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
  // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(8 : i64) : i64
  // CHECK: llvm.call @__moore_randomize_basic(%[[OBJ]], %[[SIZE]]) : (!llvm.ptr, i64) -> i32

  // Multi-range: allocate array, store range pairs, call __moore_randomize_with_ranges
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %[[ONE]] x !llvm.array<6 x i64> : (i64) -> !llvm.ptr

  // Store first range [1, 10]
  // CHECK: %[[IDX0:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %[[PTR0:.*]] = llvm.getelementptr %[[ALLOCA]][%[[IDX0]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  // CHECK: %[[MIN1:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: llvm.store %[[MIN1]], %[[PTR0]] : i64, !llvm.ptr
  // CHECK: %[[IDX1:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: %[[PTR1:.*]] = llvm.getelementptr %[[ALLOCA]][%[[IDX1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  // CHECK: %[[MAX1:.*]] = llvm.mlir.constant(10 : i64) : i64
  // CHECK: llvm.store %[[MAX1]], %[[PTR1]] : i64, !llvm.ptr

  // Store second range [20, 30]
  // CHECK: %[[IDX2:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: %[[PTR2:.*]] = llvm.getelementptr %[[ALLOCA]][%[[IDX2]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  // CHECK: %[[MIN2:.*]] = llvm.mlir.constant(20 : i64) : i64
  // CHECK: llvm.store %[[MIN2]], %[[PTR2]] : i64, !llvm.ptr
  // CHECK: %[[IDX3:.*]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK: %[[PTR3:.*]] = llvm.getelementptr %[[ALLOCA]][%[[IDX3]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  // CHECK: %[[MAX2:.*]] = llvm.mlir.constant(30 : i64) : i64
  // CHECK: llvm.store %[[MAX2]], %[[PTR3]] : i64, !llvm.ptr

  // Store third range [50, 60]
  // CHECK: %[[IDX4:.*]] = llvm.mlir.constant(4 : i64) : i64
  // CHECK: %[[PTR4:.*]] = llvm.getelementptr %[[ALLOCA]][%[[IDX4]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  // CHECK: %[[MIN3:.*]] = llvm.mlir.constant(50 : i64) : i64
  // CHECK: llvm.store %[[MIN3]], %[[PTR4]] : i64, !llvm.ptr
  // CHECK: %[[IDX5:.*]] = llvm.mlir.constant(5 : i64) : i64
  // CHECK: %[[PTR5:.*]] = llvm.getelementptr %[[ALLOCA]][%[[IDX5]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
  // CHECK: %[[MAX3:.*]] = llvm.mlir.constant(60 : i64) : i64
  // CHECK: llvm.store %[[MAX3]], %[[PTR5]] : i64, !llvm.ptr

  // Call __moore_randomize_with_ranges with the array pointer and count
  // CHECK: %[[NUM_RANGES:.*]] = llvm.mlir.constant(3 : i64) : i64
  // CHECK: %[[RANGE_RESULT:.*]] = llvm.call @__moore_randomize_with_ranges(%[[ALLOCA]], %[[NUM_RANGES]]) : (!llvm.ptr, i64) -> i64

  // Truncate to i32 and store
  // CHECK: %[[TRUNC:.*]] = arith.trunci %[[RANGE_RESULT]] : i64 to i32
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[OBJ]][0, 1]
  // CHECK: llvm.store %[[TRUNC]], %[[GEP]] : i32, !llvm.ptr

  // CHECK: %[[SUCCESS:.*]] = hw.constant true
  // CHECK: return %[[SUCCESS]] : i1
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
  // Stores range pairs [0, 50] and [100, 150]
  // CHECK-DAG: llvm.mlir.constant(0 : i64)
  // CHECK-DAG: llvm.mlir.constant(50 : i64)
  // CHECK-DAG: llvm.mlir.constant(100 : i64)
  // CHECK-DAG: llvm.mlir.constant(150 : i64)
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
  // CHECK: llvm.call @__moore_randomize_basic
  // Single range should use __moore_randomize_with_range, NOT __moore_randomize_with_ranges
  // CHECK: %[[MIN:.*]] = llvm.mlir.constant(10 : i64) : i64
  // CHECK: %[[MAX:.*]] = llvm.mlir.constant(90 : i64) : i64
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

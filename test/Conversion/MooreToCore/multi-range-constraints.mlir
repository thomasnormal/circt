// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32
// CHECK-DAG: llvm.func @__moore_is_rand_enabled(!llvm.ptr, !llvm.ptr) -> i32

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
  // Load the field value before randomization (save for restore if rand disabled)
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[OBJ]][0, 2]
  // CHECK: %[[SAVED:.*]] = llvm.load %[[GEP]] : !llvm.ptr -> i32
  // Check if rand is enabled for this field
  // CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
  // CHECK: %[[RAND_DIS:.*]] = arith.cmpi eq, {{.*}}, %false : i1
  // Call randomize_basic on the object
  // CHECK: llvm.call @__moore_randomize_basic(%[[OBJ]], {{.*}}) : (!llvm.ptr, i64) -> i32
  // CHECK: arith.trunci {{.*}} : i32 to i1
  // Restore saved value if rand was disabled
  // CHECK: scf.if %[[RAND_DIS]]
  // CHECK:   llvm.store %[[SAVED]], %[[GEP]] : i32, !llvm.ptr
  // Compute final success: basic_result AND any_rand_enabled
  // CHECK: llvm.call @__moore_is_rand_enabled
  // CHECK: arith.ori
  // CHECK: arith.andi
  // CHECK: return {{.*}} : i1
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
  // CHECK: llvm.getelementptr %[[OBJ]][0, 2]
  // CHECK: llvm.load {{.*}} : !llvm.ptr -> i32
  // CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}})
  // CHECK: llvm.call @__moore_randomize_basic(%[[OBJ]], {{.*}}) : (!llvm.ptr, i64) -> i32
  // CHECK: arith.trunci {{.*}} : i32 to i1
  // CHECK: scf.if
  // CHECK: llvm.call @__moore_is_rand_enabled
  // CHECK: arith.andi
  // CHECK: return {{.*}} : i1
  %success = moore.randomize %obj : !moore.class<@TwoRangeClass>
  return %success : i1
}

/// Test that single-range constraints use the same pattern as multi-range
/// (the old __moore_randomize_with_range API no longer exists)

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
  // CHECK: llvm.getelementptr %[[OBJ]][0, 2]
  // CHECK: llvm.load {{.*}} : !llvm.ptr -> i32
  // CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}})
  // CHECK: llvm.call @__moore_randomize_basic(%[[OBJ]], {{.*}}) : (!llvm.ptr, i64) -> i32
  // CHECK: arith.trunci {{.*}} : i32 to i1
  // CHECK: scf.if
  // Verify old API is NOT used
  // CHECK-NOT: llvm.call @__moore_randomize_with_range
  // CHECK-NOT: llvm.call @__moore_randomize_with_ranges
  // CHECK: llvm.call @__moore_is_rand_enabled
  // CHECK: arith.andi
  // CHECK: return {{.*}} : i1
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
  // Save both fields before randomization
  // CHECK: llvm.getelementptr %[[OBJ]][0, 2] {{.*}} !llvm.struct<"MixedConstraintClass", (i32, ptr, i32, i32)>
  // CHECK: llvm.load {{.*}} : !llvm.ptr -> i32
  // CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}})
  // CHECK: llvm.getelementptr %[[OBJ]][0, 3] {{.*}} !llvm.struct<"MixedConstraintClass", (i32, ptr, i32, i32)>
  // CHECK: llvm.load {{.*}} : !llvm.ptr -> i32
  // CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}})
  // Randomize the whole object
  // CHECK: llvm.call @__moore_randomize_basic(%[[OBJ]], {{.*}}) : (!llvm.ptr, i64) -> i32
  // CHECK: arith.trunci {{.*}} : i32 to i1
  // Restore fields if rand disabled
  // CHECK: scf.if
  // CHECK: scf.if
  // Final success computation
  // CHECK: llvm.call @__moore_is_rand_enabled
  // CHECK: arith.ori
  // CHECK: llvm.call @__moore_is_rand_enabled
  // CHECK: arith.ori
  // CHECK: arith.andi
  // CHECK: return {{.*}} : i1
  %success = moore.randomize %obj : !moore.class<@MixedConstraintClass>
  return %success : i1
}

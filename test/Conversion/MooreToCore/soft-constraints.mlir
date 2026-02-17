// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32
// CHECK-DAG: llvm.func @__moore_is_rand_enabled(!llvm.ptr, !llvm.ptr) -> i32

//===----------------------------------------------------------------------===//
// Soft Constraint Support Tests
// IEEE 1800-2017 Section 18.5.13 "Soft constraints"
//===----------------------------------------------------------------------===//

/// Test class with soft constraint providing default value
/// Corresponds to SystemVerilog: constraint soft_c { soft value == 42; }
/// The soft constraint sets a default value that can be overridden.

moore.class.classdecl @SoftConstraintClass {
  moore.class.propertydecl @value : !moore.i32 rand_mode rand
  moore.constraint.block @soft_c {
  ^bb0(%value: !moore.i32):
    // Soft constraint: soft value == 42 (represented as inside {[42:42]})
    moore.constraint.inside %value, [42, 42] : !moore.i32 soft
  }
}

// CHECK-LABEL: func.func @test_soft_constraint
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_soft_constraint(%obj: !moore.class<@SoftConstraintClass>) -> i1 {
  // Save old value, check rand_enabled, randomize, restore if not enabled
  // CHECK: %[[PTR:.*]] = llvm.getelementptr %[[OBJ]][0, 2]
  // CHECK: %[[OLD:.*]] = llvm.load %[[PTR]] : !llvm.ptr -> i32
  // CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
  // CHECK: llvm.call @__moore_randomize_basic(%[[OBJ]], {{.*}}) : (!llvm.ptr, i64) -> i32
  // CHECK: scf.if
  // CHECK:   llvm.store %[[OLD]], %[[PTR]] : i32, !llvm.ptr
  // CHECK: return %{{.*}} : i1
  %success = moore.randomize %obj : !moore.class<@SoftConstraintClass>
  return %success : i1
}

/// Test class with both hard and soft constraints
/// Hard constraints override soft constraints on the same property

moore.class.classdecl @HardOverridesSoftClass {
  moore.class.propertydecl @value : !moore.i32 rand_mode rand
  // Soft constraint: default to 0
  moore.constraint.block @soft_c {
  ^bb0(%value: !moore.i32):
    moore.constraint.inside %value, [0, 0] : !moore.i32 soft
  }
  // Hard constraint: must be in [10:20] - this overrides the soft constraint
  moore.constraint.block @hard_c {
  ^bb0(%value: !moore.i32):
    moore.constraint.inside %value, [10, 20] : !moore.i32
  }
}

// CHECK-LABEL: func.func @test_hard_overrides_soft
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_hard_overrides_soft(%obj: !moore.class<@HardOverridesSoftClass>) -> i1 {
  // Save old value, check rand_enabled, randomize, restore if not enabled
  // CHECK: %[[PTR:.*]] = llvm.getelementptr %[[OBJ]][0, 2]
  // CHECK: %[[OLD:.*]] = llvm.load %[[PTR]] : !llvm.ptr -> i32
  // CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
  // CHECK: llvm.call @__moore_randomize_basic(%[[OBJ]], {{.*}}) : (!llvm.ptr, i64) -> i32
  // CHECK: scf.if
  // CHECK:   llvm.store %[[OLD]], %[[PTR]] : i32, !llvm.ptr
  // CHECK: return %{{.*}} : i1
  %success = moore.randomize %obj : !moore.class<@HardOverridesSoftClass>
  return %success : i1
}

/// Test class with multiple properties - soft constraint on one, hard on another
/// Both constraints are in a single block with multiple block arguments

moore.class.classdecl @MixedConstraintsClass {
  moore.class.propertydecl @softProp : !moore.i32 rand_mode rand
  moore.class.propertydecl @hardProp : !moore.i32 rand_mode rand
  // Constraint block with both properties as block arguments
  // Block arg 0 = softProp (rand prop 0), Block arg 1 = hardProp (rand prop 1)
  moore.constraint.block @mixed_c {
  ^bb0(%softProp: !moore.i32, %hardProp: !moore.i32):
    // Soft constraint on first property (default to 100)
    moore.constraint.inside %softProp, [100, 100] : !moore.i32 soft
    // Hard constraint on second property (must be in [1:50])
    moore.constraint.inside %hardProp, [1, 50] : !moore.i32
  }
}

// CHECK-LABEL: func.func @test_mixed_constraints
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_mixed_constraints(%obj: !moore.class<@MixedConstraintsClass>) -> i1 {
  // Save old values for both properties, check rand_enabled, randomize, restore if not enabled
  // CHECK: llvm.getelementptr %[[OBJ]][0, 2]
  // CHECK: llvm.load {{.*}} : !llvm.ptr -> i32
  // CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}})
  // CHECK: llvm.getelementptr %[[OBJ]][0, 3]
  // CHECK: llvm.load {{.*}} : !llvm.ptr -> i32
  // CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}})
  // CHECK: llvm.call @__moore_randomize_basic(%[[OBJ]], {{.*}}) : (!llvm.ptr, i64) -> i32
  // CHECK: scf.if
  // CHECK:   llvm.store {{.*}} : i32, !llvm.ptr
  // CHECK: scf.if
  // CHECK:   llvm.store {{.*}} : i32, !llvm.ptr
  // CHECK: return %{{.*}} : i1
  %success = moore.randomize %obj : !moore.class<@MixedConstraintsClass>
  return %success : i1
}

/// Test AVIP-style pattern: soft noOfWaitStates == 0

moore.class.classdecl @AVIPStyleClass {
  moore.class.propertydecl @noOfWaitStates : !moore.i32 rand_mode rand
  // AVIP pattern: constraint waitState { soft noOfWaitStates == 0; }
  moore.constraint.block @waitState {
  ^bb0(%noOfWaitStates: !moore.i32):
    moore.constraint.inside %noOfWaitStates, [0, 0] : !moore.i32 soft
  }
}

// CHECK-LABEL: func.func @test_avip_soft_constraint
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_avip_soft_constraint(%obj: !moore.class<@AVIPStyleClass>) -> i1 {
  // Save old value, check rand_enabled, randomize, restore if not enabled
  // CHECK: %[[PTR:.*]] = llvm.getelementptr %[[OBJ]][0, 2]
  // CHECK: %[[OLD:.*]] = llvm.load %[[PTR]] : !llvm.ptr -> i32
  // CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
  // CHECK: llvm.call @__moore_randomize_basic(%[[OBJ]], {{.*}}) : (!llvm.ptr, i64) -> i32
  // CHECK: scf.if
  // CHECK:   llvm.store %[[OLD]], %[[PTR]] : i32, !llvm.ptr
  // CHECK: return %{{.*}} : i1
  %success = moore.randomize %obj : !moore.class<@AVIPStyleClass>
  return %success : i1
}

/// Test class with only soft constraint on a property (no hard constraint)
/// The soft constraint should be applied since nothing overrides it

moore.class.classdecl @OnlySoftClass {
  moore.class.propertydecl @defaultVal : !moore.i32 rand_mode rand
  moore.class.propertydecl @randomVal : !moore.i32 rand_mode rand
  // Only soft constraint on defaultVal
  moore.constraint.block @default_c {
  ^bb0(%defaultVal: !moore.i32):
    moore.constraint.inside %defaultVal, [999, 999] : !moore.i32 soft
  }
  // No constraint on randomVal - will be fully random
}

// CHECK-LABEL: func.func @test_only_soft
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_only_soft(%obj: !moore.class<@OnlySoftClass>) -> i1 {
  // Save old values for both properties, check rand_enabled, randomize, restore if not enabled
  // CHECK: llvm.getelementptr %[[OBJ]][0, 2]
  // CHECK: llvm.load {{.*}} : !llvm.ptr -> i32
  // CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}})
  // CHECK: llvm.getelementptr %[[OBJ]][0, 3]
  // CHECK: llvm.load {{.*}} : !llvm.ptr -> i32
  // CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}})
  // CHECK: llvm.call @__moore_randomize_basic(%[[OBJ]], {{.*}}) : (!llvm.ptr, i64) -> i32
  // CHECK: scf.if
  // CHECK:   llvm.store {{.*}} : i32, !llvm.ptr
  // CHECK: scf.if
  // CHECK:   llvm.store {{.*}} : i32, !llvm.ptr
  // CHECK: return %{{.*}} : i1
  %success = moore.randomize %obj : !moore.class<@OnlySoftClass>
  return %success : i1
}

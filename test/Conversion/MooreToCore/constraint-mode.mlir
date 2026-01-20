// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_constraint_mode_get(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @__moore_constraint_mode_set(!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK-DAG: llvm.func @__moore_constraint_mode_enable_all(!llvm.ptr) -> i32
// CHECK-DAG: llvm.func @__moore_constraint_mode_disable_all(!llvm.ptr) -> i32

moore.class.classdecl @ConstraintModeClass {
  moore.class.propertydecl @value : !moore.i32 rand_mode rand
  moore.constraint.block @c_range {
  ^bb0(%value: !moore.i32):
    moore.constraint.inside %value, [1, 5] : !moore.i32
  }
}

// CHECK-LABEL: func.func @test_constraint_mode
func.func @test_constraint_mode(%obj: !moore.class<@ConstraintModeClass>,
                                %mode: i32) -> i32 {
  // Constraint-level get.
  // CHECK: llvm.call @__moore_constraint_mode_get
  %prev0 = moore.constraint_mode %obj {constraint = @c_range}
      : !moore.class<@ConstraintModeClass> -> i32
  // Constraint-level set.
  // CHECK: llvm.call @__moore_constraint_mode_set
  %prev1 = moore.constraint_mode %obj, %mode {constraint = @c_range}
      : !moore.class<@ConstraintModeClass>, i32 -> i32
  // Class-level set (enable/disable all).
  // CHECK: llvm.call @__moore_constraint_mode_enable_all
  // CHECK: llvm.call @__moore_constraint_mode_disable_all
  %prev2 = moore.constraint_mode %obj, %mode
      : !moore.class<@ConstraintModeClass>, i32 -> i32
  return %prev1 : i32
}

// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_rand_mode_get(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @__moore_rand_mode_set(!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK-DAG: llvm.func @__moore_rand_mode_enable_all(!llvm.ptr) -> i32
// CHECK-DAG: llvm.func @__moore_rand_mode_disable_all(!llvm.ptr) -> i32

moore.class.classdecl @RandModeClass {
  moore.class.propertydecl @addr : !moore.i32 rand_mode rand
}

// CHECK-LABEL: func.func @test_rand_mode
func.func @test_rand_mode(%obj: !moore.class<@RandModeClass>,
                          %mode: !moore.i32) -> !moore.i32 {
  // Property-level get.
  // CHECK: llvm.call @__moore_rand_mode_get
  %prev0 = "moore.rand_mode"(%obj) <{property = @addr}>
      : (!moore.class<@RandModeClass>) -> !moore.i32
  // Property-level set.
  // CHECK: llvm.call @__moore_rand_mode_set
  %prev1 = "moore.rand_mode"(%obj, %mode) <{property = @addr}>
      : (!moore.class<@RandModeClass>, !moore.i32) -> !moore.i32
  // Class-level set (enable/disable all).
  // CHECK: llvm.call @__moore_rand_mode_enable_all
  // CHECK: llvm.call @__moore_rand_mode_disable_all
  %prev2 = "moore.rand_mode"(%obj, %mode)
      : (!moore.class<@RandModeClass>, !moore.i32) -> !moore.i32
  return %prev1 : !moore.i32
}

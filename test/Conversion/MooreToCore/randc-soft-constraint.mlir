// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test randc with soft constraint - randc may still be used when soft constraint is disabled
// CHECK-DAG: llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32
// CHECK-DAG: llvm.func @__moore_randc_next(!llvm.ptr, i64) -> i64

moore.class.classdecl @RandCSoftClass {
  moore.class.propertydecl @id : !moore.i8 rand_mode randc
  moore.constraint.block @soft_c {
  ^bb0(%id: !moore.i8):
    moore.constraint.inside %id, [5, 5] : !moore.i8 soft
  }
}

// CHECK-LABEL: func.func @test_randc_soft
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_randc_soft(%obj: !moore.class<@RandCSoftClass>) -> i1 {
  // When soft constraint is active, use basic randomize
  // CHECK: llvm.call @__moore_randomize_basic
  // When soft constraint is disabled, use randc
  // CHECK: llvm.call @__moore_randc_next
  %success = moore.randomize %obj : !moore.class<@RandCSoftClass>
  return %success : i1
}

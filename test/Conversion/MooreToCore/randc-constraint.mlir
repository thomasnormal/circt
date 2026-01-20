// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32
// CHECK-DAG: llvm.func @__moore_randomize_with_range(i64, i64) -> i64
// CHECK-NOT: llvm.func @__moore_randc_next

moore.class.classdecl @RandCConstrainedClass {
  moore.class.propertydecl @id : !moore.i8 rand_mode randc
  moore.constraint.block @range_c {
  ^bb0(%id: !moore.i8):
    moore.constraint.inside %id, [1, 3] : !moore.i8
  }
}

// CHECK-LABEL: func.func @test_randc_constraint
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_randc_constraint(%obj: !moore.class<@RandCConstrainedClass>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK: llvm.call @__moore_randomize_with_range
  // CHECK-NOT: llvm.call @__moore_randc_next
  %success = moore.randomize %obj : !moore.class<@RandCConstrainedClass>
  return %success : i1
}

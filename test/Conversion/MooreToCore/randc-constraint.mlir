// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32
// CHECK-DAG: llvm.func @__moore_randc_next(!llvm.ptr, i64) -> i64
// CHECK-DAG: llvm.func @__moore_is_rand_enabled(!llvm.ptr, !llvm.ptr) -> i32

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
  // Save the old value, check if rand is enabled
  // CHECK: llvm.getelementptr %[[OBJ]][0, 2]
  // CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}})
  // Call randomize_basic for the object
  // CHECK: llvm.call @__moore_randomize_basic(%[[OBJ]], {{.*}})
  // Check rand enabled again and use randc_next inside scf.if
  // CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}})
  // CHECK: scf.if
  // CHECK:   llvm.call @__moore_randc_next({{.*}})
  // Restore old value if rand was disabled
  // CHECK: scf.if
  %success = moore.randomize %obj : !moore.class<@RandCConstrainedClass>
  return %success : i1
}

// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32
// CHECK-DAG: llvm.func @__moore_randc_next(!llvm.ptr, i64) -> i64
// CHECK-DAG: llvm.func @__moore_is_rand_enabled(!llvm.ptr, !llvm.ptr) -> i32

moore.class.classdecl @RandCClass {
  moore.class.propertydecl @id : !moore.i8 rand_mode randc
}

// CHECK-LABEL: func.func @test_randc_randomize
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_randc_randomize(%obj: !moore.class<@RandCClass>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK: llvm.call @__moore_is_rand_enabled
  // CHECK: %[[FIELD_PTR:.*]] = llvm.getelementptr %[[OBJ]][0, 1]
  // CHECK: llvm.call @__moore_randc_next(%[[FIELD_PTR]]
  // CHECK: llvm.store
  %success = moore.randomize %obj : !moore.class<@RandCClass>
  return %success : i1
}

// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32
// CHECK-DAG: llvm.func @__moore_randc_next(!llvm.ptr, i64) -> i64

moore.class.classdecl @RandCMultiClass {
  moore.class.propertydecl @id0 : !moore.i8 rand_mode randc
  moore.class.propertydecl @id1 : !moore.i8 rand_mode randc
}

// CHECK-LABEL: func.func @test_randc_multi
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_randc_multi(%obj: !moore.class<@RandCMultiClass>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK: llvm.call @__moore_randc_next
  // CHECK: llvm.call @__moore_randc_next
  %success = moore.randomize %obj : !moore.class<@RandCMultiClass>
  return %success : i1
}

// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_assoc_exists(!llvm.ptr, !llvm.ptr) -> i32

moore.global_variable @testAssoc : !moore.assoc_array<!moore.i32, !moore.i8>

//===----------------------------------------------------------------------===//
// Associative Array Exists Operation
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_assoc_exists
// CHECK: [[ADDR:%.+]] = llvm.mlir.addressof @testAssoc : !llvm.ptr
// CHECK: [[ARRAY:%.+]] = llvm.load [[ADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK: [[KEY:%.+]] = hw.constant 42 : i8
// CHECK: llvm.alloca {{.*}} x i8
// CHECK: llvm.store [[KEY]], {{.*}} : i8, !llvm.ptr
// CHECK: [[CALL:%.+]] = llvm.call @__moore_assoc_exists([[ARRAY]], {{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK: comb.extract [[CALL]] from 0 : (i32) -> i1
func.func @test_assoc_exists() -> !moore.i1 {
  %assoc_ref = moore.get_global_variable @testAssoc : !moore.ref<assoc_array<!moore.i32, !moore.i8>>
  %assoc = moore.read %assoc_ref : <assoc_array<!moore.i32, !moore.i8>>
  %key = moore.constant 42 : i8
  %result = moore.assoc.exists %assoc, %key : !moore.assoc_array<!moore.i32, !moore.i8>, !moore.i8
  return %result : !moore.i1
}

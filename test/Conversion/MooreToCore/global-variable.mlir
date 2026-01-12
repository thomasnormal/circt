// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test simple integer global variable
// CHECK-LABEL: llvm.mlir.global internal @intGlobal
// CHECK-SAME: : i32
moore.global_variable @intGlobal : !moore.i32

// Test queue-typed global variable (dynamic container)
// CHECK-LABEL: llvm.mlir.global internal @queueGlobal
// CHECK-SAME: : !llvm.struct<(ptr, i64)>
moore.global_variable @queueGlobal : !moore.queue<!moore.i8, 0>

// Test dynamic array-typed global variable
// CHECK-LABEL: llvm.mlir.global internal @dynArrayGlobal
// CHECK-SAME: : !llvm.struct<(ptr, i64)>
moore.global_variable @dynArrayGlobal : !moore.open_uarray<!moore.i16>

// Test associative array-typed global variable
// CHECK-LABEL: llvm.mlir.global internal @assocArrayGlobal
// CHECK-SAME: : !llvm.ptr
moore.global_variable @assocArrayGlobal : !moore.assoc_array<!moore.i32, !moore.i8>

// Test get_global_variable for queue type
// CHECK-LABEL: func @testGetGlobalQueue
func.func @testGetGlobalQueue() {
  // CHECK: llvm.mlir.addressof @queueGlobal : !llvm.ptr
  %0 = moore.get_global_variable @queueGlobal : !moore.ref<!moore.queue<!moore.i8, 0>>
  return
}

// Test get_global_variable for integer type
// CHECK-LABEL: func @testGetGlobalInt
func.func @testGetGlobalInt() {
  // CHECK: llvm.mlir.addressof @intGlobal : !llvm.ptr
  %0 = moore.get_global_variable @intGlobal : !moore.ref<!moore.i32>
  return
}

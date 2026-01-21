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
// CHECK-SAME: () -> !llvm.ptr
func.func @testGetGlobalQueue() -> !moore.ref<queue<!moore.i8, 0>> {
  // CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @queueGlobal : !llvm.ptr
  %0 = moore.get_global_variable @queueGlobal : !moore.ref<queue<!moore.i8, 0>>
  // CHECK: return %[[ADDR]] : !llvm.ptr
  return %0 : !moore.ref<queue<!moore.i8, 0>>
}

// Test get_global_variable for integer type
// Integer ref types convert to !llhd.ref<iN> via type conversion
// CHECK-LABEL: func @testGetGlobalInt
// CHECK-SAME: () -> !llhd.ref<i32>
func.func @testGetGlobalInt() -> !moore.ref<i32> {
  // CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @intGlobal : !llvm.ptr
  %0 = moore.get_global_variable @intGlobal : !moore.ref<i32>
  // CHECK: return
  return %0 : !moore.ref<i32>
}

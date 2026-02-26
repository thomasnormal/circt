// RUN: circt-sim-compile -v %s -o %t.so 2>&1 | FileCheck %s

// CHECK: [circt-sim-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// CHECK: [circt-sim-compile] Native module init modules: 0 emitted / 1 total
// CHECK: [circt-sim-compile] Top native module init skip reasons:
// CHECK: 1x unsupported_call:puts

llvm.func @puts(!llvm.ptr) -> i32
llvm.mlir.global internal @g_data(0 : i32) : i32
llvm.mlir.global internal @g_str("skip native init\00") : !llvm.array<17 x i8>

func.func @identity(%arg0: i32) -> i32 {
  return %arg0 : i32
}

hw.module @top() {
  %g = llvm.mlir.addressof @g_data : !llvm.ptr
  %s = llvm.mlir.addressof @g_str : !llvm.ptr
  %r = llvm.call @puts(%s) : (!llvm.ptr) -> i32
  llvm.store %r, %g : i32, !llvm.ptr
  hw.output
}

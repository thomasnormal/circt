// RUN: circt-opt %s --init-vtables | FileCheck %s

// Test that InitVtablesPass populates vtable globals with function pointers
// when the referenced functions exist as llvm.func.

module {
  llvm.func @malloc(i64) -> !llvm.ptr

  // Vtable global with metadata - should be populated
  // CHECK: llvm.mlir.global internal @"TestClass::__vtable__"() {addr_space = 0 : i32} : !llvm.array<2 x ptr> {
  // CHECK:   %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.array<2 x ptr>
  // CHECK:   %[[FUNC0:.*]] = llvm.mlir.addressof @"TestClass::method0" : !llvm.ptr
  // CHECK:   %[[ARR1:.*]] = llvm.insertvalue %[[FUNC0]], %[[UNDEF]][0] : !llvm.array<2 x ptr>
  // CHECK:   %[[FUNC1:.*]] = llvm.mlir.addressof @"TestClass::method1" : !llvm.ptr
  // CHECK:   %[[ARR2:.*]] = llvm.insertvalue %[[FUNC1]], %[[ARR1]][1] : !llvm.array<2 x ptr>
  // CHECK:   llvm.return %[[ARR2]] : !llvm.array<2 x ptr>
  // CHECK: }
  llvm.mlir.global internal @"TestClass::__vtable__"(#llvm.zero) {addr_space = 0 : i32, circt.vtable_entries = [[0, @"TestClass::method0"], [1, @"TestClass::method1"]]} : !llvm.array<2 x ptr>

  // Vtable global with missing function reference - should be skipped (func.func not llvm.func)
  // CHECK: llvm.mlir.global internal @"SkippedClass::__vtable__"(#llvm.zero) {addr_space = 0 : i32, circt.vtable_entries = {{.*}}} : !llvm.array<1 x ptr>
  llvm.mlir.global internal @"SkippedClass::__vtable__"(#llvm.zero) {addr_space = 0 : i32, circt.vtable_entries = [[0, @"SkippedClass::method"]]} : !llvm.array<1 x ptr>

  // Functions as llvm.func (post func-to-llvm conversion)
  llvm.func @"TestClass::method0"(%arg0: !llvm.ptr) {
    llvm.return
  }
  llvm.func @"TestClass::method1"(%arg0: !llvm.ptr) -> i32 {
    %c42 = llvm.mlir.constant(42 : i32) : i32
    llvm.return %c42 : i32
  }

  // SkippedClass::method doesn't exist as llvm.func, so vtable won't be populated
}

// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test string type conversion to LLVM struct
// CHECK-LABEL: func.func @SimpleString
// CHECK-SAME: (%[[ARG:.*]]: !llvm.struct<(ptr, i64)>) -> !llvm.struct<(ptr, i64)>
// CHECK: return %[[ARG]] : !llvm.struct<(ptr, i64)>
func.func @SimpleString(%str: !moore.string) -> !moore.string {
  return %str : !moore.string
}

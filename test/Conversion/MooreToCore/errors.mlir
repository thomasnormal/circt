// RUN: circt-opt %s --convert-moore-to-core --split-input-file --verify-diagnostics | FileCheck %s

// Note: queue<string> is now supported, so the previous test case for
// "invalid type" was removed.

// Integer-to-string conversion is supported and lowers to the runtime string
// representation.
// CHECK-LABEL: func.func @unsupportedConversion() -> !llvm.struct<(ptr, i64)>
// CHECK: return %{{.+}} : !llvm.struct<(ptr, i64)>
func.func @unsupportedConversion() -> !moore.string {
  %0 = moore.constant_string "Test" : i32
  %1 = moore.conversion %0 : !moore.i32 -> !moore.string
  return %1 : !moore.string
}

// -----

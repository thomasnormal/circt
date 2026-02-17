// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test string type passthrough
// CHECK-LABEL: func.func @SimpleString
// CHECK-SAME: (%[[ARG:.*]]: !moore.string) -> !moore.string
// CHECK: return %[[ARG]] : !moore.string
func.func @SimpleString(%str: !moore.string) -> !moore.string {
  return %str : !moore.string
}

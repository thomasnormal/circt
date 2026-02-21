// RUN: arcilator %s --behavioral --emit-mlir | FileCheck %s

module {
  func.func @entry() -> !moore.i64 {
    %c = moore.constant 42 : !moore.i64
    return %c : !moore.i64
  }
}

// CHECK: llvm.func @entry() -> i64
// CHECK: llvm.mlir.constant(42 : i64) : i64

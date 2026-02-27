// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE

// Regression: cloned private func declarations must stay private. Otherwise
// MLIR verifier emits:
//   'func.func' op symbol declaration cannot have public visibility
//
// COMPILE: [circt-compile] Functions: 2 total, 1 external, 0 rejected, 1 compilable
// COMPILE-NOT: symbol declaration cannot have public visibility
// COMPILE: [circt-compile] 1 functions + 0 processes ready for codegen

func.func private @ext_private(%x: i32) -> i32

func.func @caller(%x: i32) -> i32 {
  %y = func.call @ext_private(%x) : (i32) -> i32
  return %y : i32
}

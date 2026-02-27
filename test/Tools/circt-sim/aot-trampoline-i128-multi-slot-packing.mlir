// RUN: circt-compile --emit-llvm %s -o %t.ll 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: FileCheck %s --check-prefix=LLVM < %t.ll

// Regression: trampoline ABI packing for integers wider than 64 bits must
// preserve all bits across slot packing/unpacking.
//
// COMPILE: [circt-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-compile] Wrote LLVM IR to
//
// LLVM: %[[ARG:[0-9]+]] = load i128, ptr %1
// LLVM: %[[RET:[0-9]+]] = tail call i128 @ext_i128(i128 %[[ARG]])
// LLVM: store i128 %[[RET]], ptr %0

func.func @entry(%x: i128) -> i128 {
  %r = llvm.call @ext_i128(%x) : (i128) -> i128
  return %r : i128
}

llvm.func @ext_i128(i128) -> i128

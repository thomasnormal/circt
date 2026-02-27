// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s

// CHECK: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// CHECK: [circt-compile] Stripped 1 functions with non-LLVM ops
// CHECK: [circt-compile] Top residual non-LLVM strip reasons:
// CHECK: 1x sig_nonllvm_arg:!hw.struct<f: i8>
// CHECK: [circt-compile] Residual stripped symbols (top 20):
// CHECK: strip_me [sig_nonllvm_arg:!hw.struct<f: i8>]

module {
  func.func private @ok() -> i32 {
    %c7_i32 = hw.constant 7 : i32
    return %c7_i32 : i32
  }

  // This function is accepted by the front-end compilability filter, but
  // carries a non-LLVM function signature that must be stripped before LLVM IR
  // translation.
  func.func private @strip_me(%arg: !hw.struct<f: i8>) -> i32 {
    %c1_i32 = hw.constant 1 : i32
    return %c1_i32 : i32
  }
}

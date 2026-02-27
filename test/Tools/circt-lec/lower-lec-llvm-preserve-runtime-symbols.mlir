// RUN: circt-opt %s --lower-lec-llvm | FileCheck %s

module {
  llvm.func @rt(i64)

  hw.module @top() {
    %c0_i64 = hw.constant 0 : i64
    llhd.process {
      llvm.call @rt(%c0_i64) : (i64) -> ()
      llhd.halt
    }
    hw.output
  }
}

// CHECK: llvm.func @rt(i64)
// CHECK: llvm.call @rt

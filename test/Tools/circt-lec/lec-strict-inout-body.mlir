// RUN: not circt-lec --emit-mlir --lec-strict -c1=top -c2=top %s %s 2>&1 | FileCheck %s

// CHECK: LEC strict mode does not support inout types; rerun without --lec-strict/--strict-llhd

// This test checks that inout port types are detected as unsupported in strict mode.
// The block argument from an inout port has type !hw.inout<i1>.

module {
  hw.module @top(inout %io : i1) {
    hw.output
  }
}

// RUN: not circt-lec --emit-mlir --strict-llhd -c1=top -c2=top %s %s 2>&1 | FileCheck %s

// CHECK: LEC strict mode does not support inout types; rerun without --lec-strict/--strict-llhd

module {
  hw.module @top(inout %io : i1) {
    hw.output
  }
}

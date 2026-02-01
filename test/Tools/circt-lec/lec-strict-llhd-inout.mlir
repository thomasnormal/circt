// RUN: circt-lec --emit-mlir --strict-llhd -c1=top -c2=top %s %s | FileCheck %s

// CHECK: smt.solver
// CHECK: smt.check

module {
  hw.module @top(inout %io : i1) {
    hw.output
  }
}

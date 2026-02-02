// RUN: circt-lec --emit-mlir --lec-strict -c1=top -c2=top %s %s | FileCheck %s

// CHECK: smt.solver
// CHECK: smt.check

module {
  hw.module @top(inout %io : i1) {
    %c0 = hw.constant 0 : i1
    %c1 = hw.constant 1 : i1
    sv.assign %io, %c0 : i1
    sv.assign %io, %c1 : i1
    hw.output
  }
}

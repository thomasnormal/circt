// RUN: circt-lec --emit-mlir --lec-strict -c1=top -c2=top %s %s | FileCheck %s

// CHECK: smt.solver
// CHECK: smt.check

module {
  hw.module @top(inout %io : i1, out o : i1) {
    %c0 = hw.constant 0 : i1
    sv.assign %io, %c0 : i1
    %r = sv.read_inout %io : !hw.inout<i1>
    hw.output %r : i1
  }
}

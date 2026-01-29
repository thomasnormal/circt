// RUN: circt-lec --emit-smtlib -c1=modA -c2=modB %s | FileCheck %s

hw.module @modA(in %a: i1, in %b: i1, out out: i1) {
  %and = comb.and %a, %b : i1
  hw.output %and : i1
}

hw.module @modB(in %a: i1, in %b: i1, out out: i1) {
  %and = comb.and %a, %b : i1
  hw.output %and : i1
}

// CHECK: (declare-const a (_ BitVec 1))
// CHECK: (declare-const b (_ BitVec 1))

// RUN: circt-lec --emit-smtlib -c1=modA -c2=modB %s | FileCheck %s

hw.module @modA(in %in: i1, out out: i1) {
  verif.assert %in label "my_assert" : i1
  hw.output %in : i1
}

hw.module @modB(in %in: i1, out out: i1) {
  hw.output %in : i1
}

// CHECK: :named my_assert

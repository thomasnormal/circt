// REQUIRES: z3
// RUN: circt-lec --run-smtlib -c1=modA -c2=modB %s | FileCheck %s

hw.module @modA(in %in: i1) {
  hw.output
}

hw.module @modB(in %in: i1) {
  hw.output
}

// CHECK: c1 == c2

// REQUIRES: z3
// RUN: circt-lec --run-smtlib --print-counterexample -c1=modA -c2=modB %s | FileCheck %s

hw.module @modA(in %in: i1, out out: i1) {
  hw.output %in : i1
}

hw.module @modB(in %in: i1, out out: i1) {
  hw.output %in : i1
}

// CHECK: c1 == c2
// CHECK: LEC_RESULT=EQ

// REQUIRES: z3
// RUN: circt-lec --emit-smtlib -c1=modA -c2=modB %s | %z3 -in | FileCheck %s

hw.module @modA(in %in: i1, out out: i1) {
  hw.output %in : i1
}

hw.module @modB(in %in: i1, out out: i1) {
  %c1 = hw.constant 1 : i1
  %0 = comb.xor %in, %c1 : i1
  hw.output %0 : i1
}

// CHECK: sat

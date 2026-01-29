// RUN: not circt-lec --run-smtlib --fail-on-inequivalent --print-solver-output \
// RUN:   --z3-path=%S/Inputs/fake-z3-model.sh -c1=modA -c2=modB %s 2>&1 \
// RUN:   | FileCheck %s

hw.module @modA(in %in: i1, out out: i1) {
  hw.output %in : i1
}

hw.module @modB(in %in: i1, out out: i1) {
  %c1 = hw.constant 1 : i1
  %0 = comb.xor %in, %c1 : i1
  hw.output %0 : i1
}

// CHECK: c1 != c2

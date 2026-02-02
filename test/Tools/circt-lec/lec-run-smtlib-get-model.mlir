// RUN: circt-lec --run-smtlib --print-counterexample \
// RUN:   --z3-path=%S/Inputs/fake-z3-model-require-get-model.sh -c1=modA -c2=modB %s 2>&1 \
// RUN:   | FileCheck %s

hw.module @modA(in %in: i1, in %in2: i8, in %in3: i72, out out: i1) {
  hw.output %in : i1
}

hw.module @modB(in %in: i1, in %in2: i8, in %in3: i72, out out: i1) {
  hw.output %in : i1
}

// CHECK: counterexample inputs:
// CHECK-DAG: in = true
// CHECK-DAG: in2 = 8'h05
// CHECK-DAG: in3 = 72'h010000000000000001
// CHECK: c1 != c2

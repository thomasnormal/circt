// RUN: circt-lec --run-smtlib --print-counterexample \
// RUN:   --z3-path=%S/Inputs/fake-z3-model-fourstate.sh -c1=modA -c2=modB %s 2>&1 \
// RUN:   | FileCheck %s

hw.module @modA(in %fs: !hw.struct<value: i8, unknown: i8>, out out: i1) {
  %false = hw.constant false
  hw.output %false : i1
}

hw.module @modB(in %fs: !hw.struct<value: i8, unknown: i8>, out out: i1) {
  %true = hw.constant true
  hw.output %true : i1
}

// CHECK: counterexample inputs:
// CHECK: fs = value=8'hB3 unknown=8'h80 (8'bz0110011, packed=16'hB380)
// CHECK: c1 != c2

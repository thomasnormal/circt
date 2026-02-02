// RUN: circt-lec --emit-smtlib -c1=modA -c2=modB %s | FileCheck %s

hw.module @modA(in %in: i1, out y: i1) {
  hw.output %in : i1
}

hw.module @modB(in %in: i1, out y: i1) {
  %c1 = hw.constant 1 : i1
  %0 = comb.xor %in, %c1 : i1
  hw.output %0 : i1
}

// CHECK-DAG: (declare-const c1_out0
// CHECK-DAG: (declare-const c2_out0
// CHECK: (= c1_out0 in)
// CHECK: (= c2_out0

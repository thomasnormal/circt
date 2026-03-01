// RUN: circt-opt --construct-lec="first-module=top_a second-module=top_b align-io=true insert-mode=none" %s | FileCheck %s

hw.module @top_a(in %a: i1, in %b: i1, out result: i1) {
  %0 = comb.xor %a, %b : i1
  hw.output %0 : i1
}

hw.module @top_b(in %b: i1, in %a: i1, out result: i1) {
  %0 = comb.xor %a, %b : i1
  hw.output %0 : i1
}

// CHECK: verif.lec {lec.input_names = ["a", "b"], lec.input_types = [i1, i1]} first {
// CHECK:   ^bb0([[A:%.+]]: i1, [[B:%.+]]: i1):
// CHECK:     [[X0:%.+]] = comb.xor [[A]], [[B]]
// CHECK:     verif.yield [[X0]]
// CHECK:   } second {
// CHECK:   ^bb0([[A2:%.+]]: i1, [[B2:%.+]]: i1):
// CHECK:     [[X1:%.+]] = comb.xor [[A2]], [[B2]]
// CHECK:     verif.yield [[X1]]
// CHECK:   }

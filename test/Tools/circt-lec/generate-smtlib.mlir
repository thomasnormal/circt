// RUN: circt-lec %s --c1 foo1 --c2 foo2 --emit-smtlib | FileCheck %s

hw.module @foo1(in %a : i8, in %b : i8, out c : i8) {
  %add = comb.add %a, %b: i8
  hw.output %add : i8
}

hw.module @foo2(in %a : i8, in %b : i8, out c : i8) {
  %add = comb.add %b, %a: i8
  hw.output %add : i8
}


// CHECK:      ; solver scope 0
// CHECK-DAG: (declare-const a (_ BitVec 8))
// CHECK-DAG: (declare-const b (_ BitVec 8))
// CHECK-DAG: (declare-const c1_out0 (_ BitVec 8))
// CHECK-DAG: (declare-const c2_out0 (_ BitVec 8))
// CHECK: (assert (let ((tmp (bvadd a b)))
// CHECK:         (let ((tmp_0 (= c1_out0 tmp)))
// CHECK:         tmp_0)))
// CHECK: (assert (let ((tmp_1 (bvadd b a)))
// CHECK:         (let ((tmp_2 (= c2_out0 tmp_1)))
// CHECK:         tmp_2)))
// CHECK: (assert (let ((tmp_3 (distinct c1_out0 c2_out0)))
// CHECK:         tmp_3))
// CHECK: (check-sat)
// CHECK: (reset)

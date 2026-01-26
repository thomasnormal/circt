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
// CHECK-NEXT: (declare-const a (_ BitVec 8))
// CHECK-NEXT: (declare-const b (_ BitVec 8))
// CHECK-NEXT: (assert (let ((tmp (bvadd b a)))
// CHECK-NEXT:         (let ((tmp_0 (bvadd a b)))
// CHECK-NEXT:         (let ((tmp_1 (distinct tmp_0 tmp)))
// CHECK-NEXT:         tmp_1))))
// CHECK-NEXT: (check-sat)
// CHECK-NEXT: (reset)

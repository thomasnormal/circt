// RUN: circt-bmc --emit-smtlib -b 1 --module top %s | FileCheck %s

// Final assertions are violated when any final check is false.
// CHECK: (declare-const a (_ BitVec 1))
// CHECK: (declare-const b (_ BitVec 1))
// CHECK: (let ((tmp{{[_0-9]*}} (or tmp{{[_0-9]*}} tmp{{[_0-9]*}})))
// CHECK: (check-sat)

hw.module @top(in %a: i1, in %b: i1) {
  verif.assert %a {bmc.final} : i1
  verif.assert %b {bmc.final} : i1
  hw.output
}

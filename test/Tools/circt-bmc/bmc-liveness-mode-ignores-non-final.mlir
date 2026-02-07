// RUN: circt-bmc --emit-smtlib -b 1 --module top --liveness %s | FileCheck %s

// Liveness mode should only check bmc.final properties.
// CHECK: (declare-const b (_ BitVec 1))
// CHECK-NOT: (declare-const a (_ BitVec 1))
// CHECK: (= b #b1)
// CHECK-NOT: (= a #b1)
// CHECK: (check-sat)

hw.module @top(in %a: i1, in %b: i1) {
  verif.assert %a : i1
  verif.assert %b {bmc.final} : i1
  hw.output
}

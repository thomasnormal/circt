// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Final assertions should violate when any one is false, not only when all are
// false.
// CHECK-LABEL: func.func @final_checks_any_violation
// CHECK: smt.push 1
// CHECK-DAG: [[NOT0:%.*]] = smt.not %{{.*}}
// CHECK-DAG: [[NOT1:%.*]] = smt.not %{{.*}}
// CHECK: [[ANY_FAIL:%.*]] = smt.or [[NOT0]], [[NOT1]]
// CHECK: smt.assert [[ANY_FAIL]]
// CHECK: smt.check
// CHECK: smt.pop 1
func.func @final_checks_any_violation() -> i1 {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1, %b: i1):
    verif.assert %a {bmc.final} : i1
    verif.assert %b {bmc.final} : i1
    verif.yield %a, %b : i1, i1
  }
  func.return %bmc : i1
}

// RUN: circt-opt %s --convert-verif-to-smt=for-smtlib-export --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: smt.solver
// CHECK: %[[NOT0:.*]] = smt.not
// CHECK: %[[NOT1:.*]] = smt.not
// CHECK: %[[OR:.*]] = smt.or %[[NOT0]], %[[NOT1]]
// CHECK: %[[ORW:.*]] = smt.or
// CHECK: smt.assert %[[ORW]]

func.func @bmc_final_checks_smtlib() -> i1 {
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

// RUN: not circt-opt %s --convert-verif-to-smt='bmc-mode=liveness-lasso' --reconcile-unrealized-casts -allow-unregistered-dialect 2>&1 | FileCheck %s

// CHECK: error: liveness-lasso mode currently requires SMT-LIB export (use --emit-smtlib or --run-smtlib)
func.func @liveness_lasso_requires_smtlib() -> i1 {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1):
    verif.assert %a {bmc.final} : i1
    verif.yield %a : i1
  }
  func.return %bmc : i1
}

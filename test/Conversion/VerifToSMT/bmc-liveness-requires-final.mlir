// RUN: not circt-opt %s --convert-verif-to-smt='bmc-mode=liveness' --reconcile-unrealized-casts -allow-unregistered-dialect 2>&1 | FileCheck %s

// CHECK: error: liveness mode requires at least one bmc.final property
func.func @liveness_requires_final() -> i1 {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1):
    verif.assert %a : i1
    verif.yield %a : i1
  }
  func.return %bmc : i1
}

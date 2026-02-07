// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true bmc-mode=induction-step' --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// In induction-step mode, final checks are now assumed to hold on all prefix
// iterations (k-window) before checking the final step.
// CHECK-LABEL: func.func @induction_final_prefix_assume
// CHECK-COUNT-2: smt.assert
func.func @induction_final_prefix_assume() -> i1 {
  %bmc = verif.bmc bound 2 num_regs 0 initial_values []
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

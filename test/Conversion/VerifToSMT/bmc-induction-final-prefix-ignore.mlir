// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true bmc-mode=induction-step' --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// In induction-step mode, ignore_asserts_until skips the early prefix-window
// final assumptions. With bound=2 and ignore=1, only the final-step check
// remains.
// CHECK-LABEL: func.func @induction_final_prefix_ignore
// CHECK-COUNT-1: smt.assert
func.func @induction_final_prefix_ignore() -> i1 {
  %bmc = verif.bmc bound 2 num_regs 0 initial_values [] attributes {ignore_asserts_until = 1 : i64}
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

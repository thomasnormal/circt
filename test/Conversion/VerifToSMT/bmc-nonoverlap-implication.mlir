// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @bmc_circuit
// Delay buffer should be sourced from the antecedent after shift-rewrite.
// Returns: orig outputs + delay buffer + non-final check (!smt.bool)
// CHECK: return %arg0, %arg1, %arg0, %{{.*}} : !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bool
func.func @test_nonoverlap() -> i1 {
  %bmc = verif.bmc bound 3 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1, %b: i1):
    %d = ltl.delay %b, 1, 0 : i1
    %p = ltl.implication %a, %d : i1, !ltl.sequence
    verif.assert %p : !ltl.property
    verif.yield %a, %b : i1, i1
  }
  func.return %bmc : i1
}

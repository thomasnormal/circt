// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Test multi-step BMC expansion for goto/non-consecutive repetition.

// =============================================================================
// Test case 1: goto repeat (exactly 2 occurrences, last at current cycle)
// =============================================================================

// CHECK-LABEL: func.func @test_goto_repeat_2
// Single delay buffer constant initialized to false (used multiple times)
// CHECK:         {{(arith|smt)\.constant}} false
// CHECK:         scf.for
// NFA-based multi-step tracking adds a tick input as the 2nd argument.
// CHECK:           %[[TICK:.+]] = smt.ite %true, %c-1_bv1, %c0_bv1 : !smt.bv<1>
// CHECK:           func.call @bmc_circuit(%arg1, %[[TICK]],
// CHECK-SAME:        -> ({{.*}}, !smt.bool)
func.func @test_goto_repeat_2() -> i1 {
  %bmc = verif.bmc bound 4 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1):
    %seq = ltl.goto_repeat %a, 2, 0 : i1
    verif.assert %seq : !ltl.sequence
    verif.yield %a : i1
  }
  func.return %bmc : i1
}

// =============================================================================
// Test case 2: non-consecutive repeat (1 to 2 occurrences)
// =============================================================================

// CHECK-LABEL: func.func @test_nonconsecutive_repeat_range
// Single delay buffer constant initialized to false (used multiple times)
// CHECK:         {{(arith|smt)\.constant}} false
// CHECK:         scf.for
// NFA-based multi-step tracking adds a tick input as the 2nd argument.
// CHECK:           %[[TICK0:.+]] = smt.ite %true, %c-1_bv1, %c0_bv1 : !smt.bv<1>
// CHECK:           func.call @bmc_circuit_0(%arg1, %[[TICK0]],
// CHECK-SAME:        -> ({{.*}}, !smt.bool)
func.func @test_nonconsecutive_repeat_range() -> i1 {
  %bmc = verif.bmc bound 3 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1):
    %seq = ltl.non_consecutive_repeat %a, 1, 1 : i1
    verif.assert %seq : !ltl.sequence
    verif.yield %a : i1
  }
  func.return %bmc : i1
}

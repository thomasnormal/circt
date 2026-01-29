// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Test multi-step BMC expansion for goto/non-consecutive repetition.

// =============================================================================
// Test case 1: goto repeat (exactly 2 occurrences, last at current cycle)
// =============================================================================

// CHECK-LABEL: func.func @test_goto_repeat_2
// Single delay buffer constant initialized to false (used multiple times)
// CHECK:         smt.constant false
// CHECK:         scf.for
// Circuit takes 4 args (a, buf1, buf2, buf3)
// Returns: orig outputs + delay buffers + non-final check
// CHECK:           func.call @bmc_circuit
// CHECK-SAME:        : (!smt.bv<1>, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.bool) -> (!smt.bv<1>, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.bool, !smt.bool)
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
// CHECK:         smt.constant false
// CHECK:         scf.for
// Circuit takes 3 args (a, buf1, buf2)
// Returns: orig outputs + delay buffers + non-final check
// CHECK:           func.call @bmc_circuit
// CHECK-SAME:        : (!smt.bv<1>, !smt.bool, !smt.bool, !smt.bool) -> (!smt.bv<1>, !smt.bool, !smt.bool, !smt.bool, !smt.bool)
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

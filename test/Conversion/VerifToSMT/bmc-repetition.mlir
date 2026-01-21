// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Test BMC handling of SVA consecutive repetition patterns [*N] and [*M:N].
// These patterns specify that a signal must hold for N consecutive cycles
// or for a range of M to N consecutive cycles.

// =============================================================================
// Test case 1: Basic exact repetition [*3] - signal holds for 3 consecutive cycles
// =============================================================================

// The repeat operation ltl.repeat %a, 3, 0 means a[*3]
// This should be expanded to: a && ##1 a && ##2 a
// Which requires delay buffers for offsets 1 and 2

// CHECK-LABEL: func.func @test_exact_repeat_3
// Delay buffer constant initialized to 0 (single constant, used multiple times)
// CHECK: smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK: scf.for
// Circuit takes signal 'a' plus 3 delay buffer slots
// CHECK: func.call @bmc_circuit
func.func @test_exact_repeat_3() -> i1 {
  %bmc = verif.bmc bound 5 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1):
    // a[*3] - signal 'a' must hold for 3 consecutive cycles
    %rep = ltl.repeat %a, 3, 0 : i1
    verif.assert %rep : !ltl.sequence
    verif.yield %a : i1
  }
  func.return %bmc : i1
}

// =============================================================================
// Test case 2: Range repetition [*1:3] - signal holds for 1 to 3 cycles
// =============================================================================

// The repeat operation ltl.repeat %a, 1, 2 means a[*1:3]
// This is equivalent to: a || (a && ##1 a) || (a && ##1 a && ##2 a)
// Which requires delay buffers for offsets 1 and 2

// CHECK-LABEL: func.func @test_range_repeat_1_to_3
// Delay buffer constant initialized to 0 (single constant, used multiple times)
// CHECK: smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK: scf.for
// Circuit takes signal 'a' plus delay buffer slots
// CHECK: func.call @bmc_circuit
func.func @test_range_repeat_1_to_3() -> i1 {
  %bmc = verif.bmc bound 5 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1):
    // a[*1:3] - signal 'a' must hold for 1 to 3 consecutive cycles
    %rep = ltl.repeat %a, 1, 2 : i1
    verif.assert %rep : !ltl.sequence
    verif.yield %a : i1
  }
  func.return %bmc : i1
}

// =============================================================================
// Test case 3: Zero-or-more repetition [*0:$] approximated as [*0:bound]
// =============================================================================

// The repeat operation ltl.repeat %a, 0 means a[*0:$] (unbounded)
// For BMC, this is approximated within the bound as a[*0:bound-1]

// CHECK-LABEL: func.func @test_zero_or_more
// CHECK: scf.for
// Circuit returns output + non-final check
// CHECK: func.call @bmc_circuit
// CHECK-SAME: -> ({{.*}}!smt.bool)
func.func @test_zero_or_more() -> i1 {
  %bmc = verif.bmc bound 4 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1):
    // a[*0:$] - signal 'a' can hold for 0 or more cycles
    // Approximated by BMC bound
    %rep = ltl.repeat %a, 0 : i1
    verif.assert %rep : !ltl.sequence
    verif.yield %a : i1
  }
  func.return %bmc : i1
}

// =============================================================================
// Test case 4: Repetition in antecedent - a[*3] |-> b
// =============================================================================

// When the repetition is in the antecedent of an implication,
// it means "if 'a' has held for 3 consecutive cycles, then 'b' must hold"

// CHECK-LABEL: func.func @test_repeat_in_antecedent
// CHECK: scf.for
// Circuit returns outputs + non-final check
// CHECK: func.call @bmc_circuit
// CHECK-SAME: -> ({{.*}}!smt.bool)
func.func @test_repeat_in_antecedent() -> i1 {
  %bmc = verif.bmc bound 5 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1, %b: i1):
    // a[*3] |-> b - if 'a' holds for 3 consecutive cycles, 'b' must hold
    %rep = ltl.repeat %a, 3, 0 : i1
    %prop = ltl.implication %rep, %b : !ltl.sequence, i1
    verif.assert %prop : !ltl.property
    verif.yield %a, %b : i1, i1
  }
  func.return %bmc : i1
}

// =============================================================================
// Test case 5: Repetition with consequent - a |-> b[*2]
// =============================================================================

// When the repetition is in the consequent of an implication,
// it means "when 'a' holds, 'b' must hold for 2 consecutive cycles"

// CHECK-LABEL: func.func @test_repeat_in_consequent
// CHECK: scf.for
// Circuit returns outputs + non-final check
// CHECK: func.call @bmc_circuit
// CHECK-SAME: -> ({{.*}}!smt.bool)
func.func @test_repeat_in_consequent() -> i1 {
  %bmc = verif.bmc bound 5 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1, %b: i1):
    // a |-> b[*2] - when 'a' holds, 'b' must hold for 2 consecutive cycles
    %rep = ltl.repeat %b, 2, 0 : i1
    %prop = ltl.implication %a, %rep : i1, !ltl.sequence
    verif.assert %prop : !ltl.property
    verif.yield %a, %b : i1, i1
  }
  func.return %bmc : i1
}

// =============================================================================
// Test case 6: Single repetition [*1] - degenerate case (just the signal)
// =============================================================================

// The repeat operation ltl.repeat %a, 1, 0 means a[*1]
// This is equivalent to just 'a' with no delay buffers needed

// CHECK-LABEL: func.func @test_single_repeat
// No extra delay buffers needed for [*1]
// CHECK: scf.for
// Circuit should only take signal 'a' with no delay buffers
// CHECK: func.call @bmc_circuit
func.func @test_single_repeat() -> i1 {
  %bmc = verif.bmc bound 5 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1):
    // a[*1] - signal 'a' must hold for exactly 1 cycle
    %rep = ltl.repeat %a, 1, 0 : i1
    verif.assert %rep : !ltl.sequence
    verif.yield %a : i1
  }
  func.return %bmc : i1
}

// =============================================================================
// Test case 7: Zero repetition [*0] - empty sequence (trivially true)
// =============================================================================

// The repeat operation ltl.repeat %a, 0, 0 means a[*0]
// An empty sequence is trivially true

// CHECK-LABEL: func.func @test_zero_repeat
// No delay buffers needed - empty repetition is trivially true
// CHECK: scf.for
// CHECK: func.call @bmc_circuit
func.func @test_zero_repeat() -> i1 {
  %bmc = verif.bmc bound 5 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1):
    // a[*0] - empty sequence, trivially true
    %rep = ltl.repeat %a, 0, 0 : i1
    verif.assert %rep : !ltl.sequence
    verif.yield %a : i1
  }
  func.return %bmc : i1
}

// =============================================================================
// Test case 8: Combination - a[*2] ##1 b[*2]
// =============================================================================

// This tests concatenation of repeated sequences with a delay

// CHECK-LABEL: func.func @test_repeat_concat
// CHECK: scf.for
// Circuit returns outputs + non-final check
// CHECK: func.call @bmc_circuit
// CHECK-SAME: -> ({{.*}}!smt.bool)
func.func @test_repeat_concat() -> i1 {
  %bmc = verif.bmc bound 8 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1, %b: i1):
    // a[*2] - 'a' holds for 2 cycles
    %rep_a = ltl.repeat %a, 2, 0 : i1
    // b[*2] - 'b' holds for 2 cycles
    %rep_b = ltl.repeat %b, 2, 0 : i1
    // ##1 b[*2] - after 1 cycle delay, 'b' holds for 2 cycles
    %delay_rep_b = ltl.delay %rep_b, 1, 0 : !ltl.sequence
    // a[*2] ##1 b[*2] - concatenation
    %concat = ltl.concat %rep_a, %delay_rep_b : !ltl.sequence, !ltl.sequence
    verif.assert %concat : !ltl.sequence
    verif.yield %a, %b : i1, i1
  }
  func.return %bmc : i1
}

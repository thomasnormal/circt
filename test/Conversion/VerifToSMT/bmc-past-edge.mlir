// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Test multi-step BMC with ltl.past for edge detection ($rose/$fell).
// $rose(x) is represented as x && !past(x, 1)
// $fell(x) is represented as !x && past(x, 1)
// The past buffer infrastructure tracks signal history across time steps.

// =============================================================================
// Test case 1: Simple ltl.past tracking
// Verifies that past buffer infrastructure is set up correctly
// =============================================================================

// CHECK-LABEL: func.func @test_past_simple
// The past buffer is initialized to 0
// CHECK:         %[[PAST_INIT:.*]] = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// The for loop has the past buffer as an iter_arg
// CHECK:         scf.for
// Circuit is called with past buffer argument (2 total args: sig, past_buffer)
// Returns: orig outputs + past buffer + non-final check (!smt.bool)
// CHECK:           func.call @bmc_circuit
// CHECK-SAME:        : (!smt.bv<1>, !smt.bv<1>) -> (!smt.bv<1>, !smt.bv<1>, !smt.bool)
func.func @test_past_simple() -> i1 {
  %bmc = verif.bmc bound 5 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%sig: i1):
    // past(sig, 1) gives the value of sig from the previous cycle
    %past_sig = ltl.past %sig, 1 : i1
    // Assert the past value to trigger BMC checking
    verif.assert %past_sig : !ltl.sequence
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}

// =============================================================================
// Test case 2: $rose pattern ($rose(sig) = sig && !past(sig, 1))
// =============================================================================

// CHECK-LABEL: func.func @test_rose_pattern
// CHECK:         scf.for
// CHECK:           func.call @bmc_circuit
func.func @test_rose_pattern() -> i1 {
  %bmc = verif.bmc bound 5 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%sig: i1):
    // $rose(sig) = sig && !past(sig, 1)
    %past_sig = ltl.past %sig, 1 : i1
    %not_past = ltl.not %past_sig : !ltl.sequence
    %rose = ltl.and %sig, %not_past : i1, !ltl.property
    verif.assert %rose : !ltl.property
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}

// =============================================================================
// Test case 3: Multiple past delays (past(sig,1) and past(sig,2))
// Each requires its own buffer slots
// =============================================================================

// CHECK-LABEL: func.func @test_multiple_past
// Single buffer constant (used multiple times for past(sig,1) + past(sig,2))
// CHECK:         smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK:         scf.for
// Returns: orig outputs + past buffers + non-final check
// CHECK:           func.call @bmc_circuit
// CHECK-SAME:        : (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>) -> (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bool)
func.func @test_multiple_past() -> i1 {
  %bmc = verif.bmc bound 5 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%sig: i1):
    // past(sig, 1) - needs 1 buffer slot
    %past1 = ltl.past %sig, 1 : i1
    // past(sig, 2) - needs 2 buffer slots
    %past2 = ltl.past %sig, 2 : i1
    // Assert the conjunction
    %and = ltl.and %past1, %past2 : !ltl.sequence, !ltl.sequence
    verif.assert %and : !ltl.sequence
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}

// =============================================================================
// Test case 4: Combining past with delay (for patterns like $rose |-> ##1 ack)
// Both past buffers and delay buffers should work together
// =============================================================================

// CHECK-LABEL: func.func @test_past_with_delay
// Both delay and past buffers are allocated
// CHECK:         scf.for
// Circuit has 4 args: req, ack, delay_buffer, past_buffer
// Returns: orig outputs + buffers + 2 non-final checks (one per assert)
// CHECK:           func.call @bmc_circuit
// CHECK-SAME:        : (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>) -> (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bool, !smt.bool)
func.func @test_past_with_delay() -> i1 {
  %bmc = verif.bmc bound 5 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%req: i1, %ack: i1):
    // past(req, 1) for edge detection
    %past_req = ltl.past %req, 1 : i1
    // ##1 ack for delayed consequent
    %delayed_ack = ltl.delay %ack, 1, 0 : i1
    // req |-> ##1 ack
    %prop = ltl.implication %req, %delayed_ack : i1, !ltl.sequence
    verif.assert %prop : !ltl.property
    // Also assert past to ensure both buffers work
    verif.assert %past_req : !ltl.sequence
    verif.yield %req, %ack : i1, i1
  }
  func.return %bmc : i1
}

// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Test multi-step BMC with ltl.delay for temporal properties.
// This tests the delay buffer infrastructure for tracking delayed obligations
// across time steps in bounded model checking.

// =============================================================================
// Test case 1: Simple ##1 delay (1-cycle delay)
// Property: req |-> ##1 ack
// Meaning: When req is high, ack must be high in the NEXT cycle
// =============================================================================

// CHECK-LABEL: func.func @test_delay_1
// The delay buffer is initialized to 0
// CHECK:         %[[DELAY_INIT:.*]] = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// The for loop has the delay buffer as an iter_arg
// CHECK:         scf.for
// Circuit is called with 3 args (req, ack, delay_buffer)
// Returns: orig outputs + delay buffer + non-final check (!smt.bool for LTL property)
// CHECK:           func.call @bmc_circuit
// CHECK-SAME:        : (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>) -> (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bool)
func.func @test_delay_1() -> i1 {
  %bmc = verif.bmc bound 5 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%req: i1, %ack: i1):
    // Create delayed sequence: ##1 ack
    %delayed_ack = ltl.delay %ack, 1, 0 : i1
    // Create implication: req |-> ##1 ack
    %prop = ltl.implication %req, %delayed_ack : i1, !ltl.sequence
    verif.assert %prop : !ltl.property
    verif.yield %req, %ack : i1, i1
  }
  func.return %bmc : i1
}

// =============================================================================
// Test case 2: ##2 delay (2-cycle delay)
// For delay N=2, we need 2 buffer slots
// =============================================================================

// CHECK-LABEL: func.func @test_delay_2
// Single delay buffer constant initialized to 0 (used multiple times)
// CHECK:         smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK:         scf.for
// Circuit takes 4 args (start, done, buf0, buf1)
// Returns: orig outputs + delay buffers + non-final check
// CHECK:           func.call @bmc_circuit
// CHECK-SAME:        : (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>) -> (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bool)
func.func @test_delay_2() -> i1 {
  %bmc = verif.bmc bound 5 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%start: i1, %done: i1):
    %delayed_done = ltl.delay %done, 2, 0 : i1
    %prop = ltl.implication %start, %delayed_done : i1, !ltl.sequence
    verif.assert %prop : !ltl.property
    verif.yield %start, %done : i1, i1
  }
  func.return %bmc : i1
}

// =============================================================================
// Test case 2b: ##[1:3] delay range (1 to 3 cycles)
// For delay=1 length=2, we need 3 buffer slots
// =============================================================================

// CHECK-LABEL: func.func @test_delay_range
// Single delay buffer constant initialized to 0 (used multiple times)
// CHECK:         smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK:         scf.for
// Circuit takes 5 args (start, done, buf0, buf1, buf2)
// Returns: orig outputs + delay buffers + non-final check
// CHECK:           func.call @bmc_circuit
// CHECK-SAME:        : (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>) -> (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bool)
func.func @test_delay_range() -> i1 {
  %bmc = verif.bmc bound 5 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%start: i1, %done: i1):
    %delayed_done = ltl.delay %done, 1, 2 : i1
    %prop = ltl.implication %start, %delayed_done : i1, !ltl.sequence
    verif.assert %prop : !ltl.property
    verif.yield %start, %done : i1, i1
  }
  func.return %bmc : i1
}

// =============================================================================
// Test case 2c: ##[1:$] unbounded delay (truncated by BMC bound)
// For bound=5, delay=1 length=3, we need 4 buffer slots
// =============================================================================

// CHECK-LABEL: func.func @test_delay_unbounded
// Single delay buffer constant initialized to 0 (used multiple times)
// CHECK:         smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK:         scf.for
// Circuit takes 6 args (start, done, buf0, buf1, buf2, buf3)
// Returns: orig outputs + delay buffers + non-final check
// CHECK:           func.call @bmc_circuit
// CHECK-SAME:        : (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>) -> (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bool)
func.func @test_delay_unbounded() -> i1 {
  %bmc = verif.bmc bound 5 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%start: i1, %done: i1):
    %delayed_done = ltl.delay %done, 1 : i1
    %prop = ltl.implication %start, %delayed_done : i1, !ltl.sequence
    verif.assert %prop : !ltl.property
    verif.yield %start, %done : i1, i1
  }
  func.return %bmc : i1
}

// =============================================================================
// Test case 2d: ##[0:2] delay range (0 to 2 cycles)
// For delay=0 length=2, we need 2 buffer slots
// =============================================================================

// CHECK-LABEL: func.func @test_delay_range_zero
// Single delay buffer constant initialized to 0 (used multiple times)
// CHECK:         smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK:         scf.for
// Circuit takes 4 args (start, done, buf0, buf1)
// Returns: orig outputs + delay buffers + non-final check
// CHECK:           func.call @bmc_circuit
// CHECK-SAME:        : (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>) -> (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bool)
func.func @test_delay_range_zero() -> i1 {
  %bmc = verif.bmc bound 5 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%start: i1, %done: i1):
    %delayed_done = ltl.delay %done, 0, 2 : i1
    %prop = ltl.implication %start, %delayed_done : i1, !ltl.sequence
    verif.assert %prop : !ltl.property
    verif.yield %start, %done : i1, i1
  }
  func.return %bmc : i1
}

// =============================================================================
// Test case 2e: ##[0:2] delay range on ltl.sequence input
// Ensures bounded delay ranges handle sequence-typed buffers.
// =============================================================================

// CHECK-LABEL: func.func @test_delay_range_sequence
// CHECK:         scf.for
// CHECK:           func.call @bmc_circuit
// CHECK-SAME:        : (!smt.bv<1>, !smt.bool, !smt.bool) -> (!smt.bv<1>, !smt.bool, !smt.bool, !smt.bool)
func.func @test_delay_range_sequence() -> i1 {
  %bmc = verif.bmc bound 5 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%sig: i1):
    %seq = ltl.delay %sig, 0, 0 : i1
    %delayed_seq = ltl.delay %seq, 0, 2 : !ltl.sequence
    %prop = ltl.implication %sig, %delayed_seq : i1, !ltl.sequence
    verif.assert %prop : !ltl.property
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}

// =============================================================================
// Test case 3: No delay ops - with assert (verifies non-delay paths work)
// =============================================================================

// CHECK-LABEL: func.func @test_no_delay
// Should work without delay buffers - just 2 symbolic inputs + wasViolated
// CHECK:         scf.for
// Returns: orig outputs + non-final check (i1 property -> !smt.bv<1>)
// CHECK:           func.call @bmc_circuit
// CHECK-SAME:        : (!smt.bv<1>, !smt.bv<1>) -> (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>)
func.func @test_no_delay() -> i1 {
  %bmc = verif.bmc bound 5 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1, %b: i1):
    %eq = comb.icmp eq %a, %b : i1
    verif.assert %eq : i1  // Use assert instead of assume to trigger BMC
    verif.yield %a, %b : i1, i1
  }
  func.return %bmc : i1
}

// =============================================================================
// Test case 4: delay(prop, 0) should be a pass-through (no buffer)
// =============================================================================

// CHECK-LABEL: func.func @test_delay_0
// CHECK:         scf.for
// Zero delay doesn't add buffer slots - circuit has only 2 args
// Returns: orig outputs + non-final check (!ltl.sequence -> !smt.bool)
// CHECK:           func.call @bmc_circuit
// CHECK-SAME:        : (!smt.bv<1>, !smt.bv<1>) -> (!smt.bv<1>, !smt.bv<1>, !smt.bool)
func.func @test_delay_0() -> i1 {
  %bmc = verif.bmc bound 5 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1, %b: i1):
    // delay 0 should just pass through
    %no_delay = ltl.delay %b, 0, 0 : i1
    verif.assert %no_delay : !ltl.sequence
    verif.yield %a, %b : i1, i1
  }
  func.return %bmc : i1
}

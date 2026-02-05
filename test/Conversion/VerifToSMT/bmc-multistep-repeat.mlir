// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Test multi-step BMC repeat expansion using delay buffers.

// =============================================================================
// Test case 1: exact repeat (3 consecutive cycles)
// =============================================================================

// CHECK-LABEL: func.func @test_repeat_3
// Single delay buffer constant initialized to 0 (used multiple times)
// CHECK: smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK: scf.for
// CHECK: func.call @bmc_circuit
// CHECK-SAME: -> ({{.*}}!smt.bool)
func.func @test_repeat_3() -> i1 {
  %bmc = verif.bmc bound 5 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1):
    %rep = ltl.repeat %a, 3, 0 : i1
    %prop = ltl.implication %a, %rep : i1, !ltl.sequence
    verif.assert %prop : !ltl.property
    verif.yield %a : i1
  }
  func.return %bmc : i1
}

// =============================================================================
// Test case 2: bounded repeat range (2 or 3 consecutive cycles)
// =============================================================================

// CHECK-LABEL: func.func @test_repeat_range
// Single delay buffer constant initialized to 0 (used multiple times)
// CHECK: smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK: scf.for
// CHECK: func.call @bmc_circuit
// CHECK-SAME: -> ({{.*}}!smt.bool)
func.func @test_repeat_range() -> i1 {
  %bmc = verif.bmc bound 5 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1):
    %rep = ltl.repeat %a, 2, 1 : i1
    %prop = ltl.implication %a, %rep : i1, !ltl.sequence
    verif.assert %prop : !ltl.property
    verif.yield %a : i1
  }
  func.return %bmc : i1
}

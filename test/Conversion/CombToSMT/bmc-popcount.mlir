// RUN: circt-opt %s --convert-hw-to-smt --convert-comb-to-smt --convert-verif-to-smt --reconcile-unrealized-casts | FileCheck %s

// Test BMC with popcount (llvm.intr.ctpop) operations.
// This tests that $countones, $onehot, and $onehot0 can be verified with BMC.
// These functions are lowered from Moore to Core using LLVM's ctpop intrinsic.

// =============================================================================
// Test case 1: $countones (popcount) verification
// Verify that the SMT lowering of ctpop works correctly in a BMC context.
// =============================================================================

// The BMC circuit function should contain the expanded popcount operations
// CHECK-LABEL: func.func @bmc_circuit
// CHECK:   smt.bv.extract
// CHECK:   smt.bv.concat
// CHECK:   smt.bv.add
func.func @test_popcount_bmc() -> i1 {
  %bmc = verif.bmc bound 3 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%sel: i4):
    // Compute popcount of sel
    %count = "llvm.intr.ctpop"(%sel) : (i4) -> i4

    // Check if popcount is in range [0, 4] (always true for 4-bit input)
    %c5 = hw.constant 5 : i4
    %valid = comb.icmp ult %count, %c5 : i4

    verif.assert %valid : i1
    verif.yield
  }
  func.return %bmc : i1
}

// =============================================================================
// Test case 2: $onehot pattern (popcount == 1)
// This is the pattern generated for $onehot(x) in SystemVerilog.
// =============================================================================

// CHECK-LABEL: func.func @bmc_circuit_0
// CHECK:   smt.bv.extract
// CHECK:   smt.bv.concat
// CHECK:   smt.bv.add
// CHECK:   smt.eq
func.func @test_onehot_bmc() -> i1 {
  %bmc = verif.bmc bound 3 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%sel: i4):
    // $onehot(sel) = popcount(sel) == 1
    %count = "llvm.intr.ctpop"(%sel) : (i4) -> i4
    %c1 = hw.constant 1 : i4
    %is_onehot = comb.icmp eq %count, %c1 : i4

    // We can't assert $onehot always holds (it doesn't for arbitrary input)
    // Instead, verify that if $onehot holds, some property follows
    // Here we verify: $onehot(sel) implies sel != 0
    %c0 = hw.constant 0 : i4
    %sel_nonzero = comb.icmp ne %sel, %c0 : i4

    // implication: !is_onehot || sel_nonzero
    %c_neg1 = hw.constant -1 : i1
    %not_onehot = comb.xor %is_onehot, %c_neg1 : i1
    %implication = comb.or %not_onehot, %sel_nonzero : i1

    verif.assert %implication : i1
    verif.yield
  }
  func.return %bmc : i1
}

// =============================================================================
// Test case 3: $onehot0 pattern (popcount <= 1)
// This is the pattern generated for $onehot0(x) in SystemVerilog.
// =============================================================================

// CHECK-LABEL: func.func @bmc_circuit_1
// CHECK:   smt.bv.extract
// CHECK:   smt.bv.add
// CHECK:   smt.bv.cmp ule
func.func @test_onehot0_bmc() -> i1 {
  %bmc = verif.bmc bound 3 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%sel: i4):
    // $onehot0(sel) = popcount(sel) <= 1
    %count = "llvm.intr.ctpop"(%sel) : (i4) -> i4
    %c1 = hw.constant 1 : i4
    %is_onehot0 = comb.icmp ule %count, %c1 : i4

    // Verify: $onehot0(sel) implies popcount(sel) < 2
    %c2 = hw.constant 2 : i4
    %count_lt_2 = comb.icmp ult %count, %c2 : i4

    // implication: !is_onehot0 || count_lt_2
    %c_neg1 = hw.constant -1 : i1
    %not_onehot0 = comb.xor %is_onehot0, %c_neg1 : i1
    %implication = comb.or %not_onehot0, %count_lt_2 : i1

    verif.assert %implication : i1
    verif.yield
  }
  func.return %bmc : i1
}

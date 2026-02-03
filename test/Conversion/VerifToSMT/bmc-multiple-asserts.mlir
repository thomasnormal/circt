// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Test that multiple verif.assert ops in a BMC circuit are returned separately
// and combined in the main loop using smt.and/smt.or for property checking.
// Each assert becomes a separate !smt.bv<1> return value from bmc_circuit.

// CHECK-LABEL: func.func @test_multi_asserts
// CHECK:         scf.for
// Loop is called first
// CHECK:           func.call @bmc_loop
// Circuit returns original outputs + one !smt.bv<1> per assert
// CHECK:           func.call @bmc_circuit
// CHECK-SAME:        -> (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>)
// Properties are checked separately and combined with smt.or
// CHECK:           smt.eq {{%.+}}, {{%.+}} : !smt.bv<1>
// CHECK:           smt.not
// CHECK:           smt.eq {{%.+}}, {{%.+}} : !smt.bv<1>
// CHECK:           smt.not
// CHECK:           smt.or
// CHECK:           smt.push 1
// CHECK:           smt.assert
// CHECK:           smt.check
// CHECK:           smt.pop 1
// CHECK-LABEL: func.func @bmc_circuit
// Each assert contributes a separate return value (no smt.and combining)
// CHECK-SAME:    -> (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>)
func.func @test_multi_asserts() -> i1 {
  %bmc = verif.bmc bound 2 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1, %b: i1):
    %eq = comb.icmp eq %a, %b : i1
    %neq = comb.icmp ne %a, %b : i1
    verif.assert %eq : i1
    verif.assert %neq : i1
    verif.yield %a, %b : i1, i1
  }
  func.return %bmc : i1
}

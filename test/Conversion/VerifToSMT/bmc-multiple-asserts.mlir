// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @test_multi_asserts
// Multiple asserts violation is checked as smt.or (any violated = failure)
// CHECK: smt.or
// CHECK-LABEL: func.func @bmc_circuit
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

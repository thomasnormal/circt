// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @bmc_circuit
// Multiple asserts are combined with smt.and and returned as output
// CHECK: smt.and
// CHECK: return
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

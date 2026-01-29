// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @test_final_checks
// CHECK: scf.for
// CHECK: func.call @bmc_circuit
// CHECK-SAME: -> (!smt.bv<1>, !smt.bool, !smt.bool)
// CHECK: smt.check
// CHECK: smt.check
func.func @test_final_checks() -> i1 {
  %bmc = verif.bmc bound 4 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%sig: i1):
    %sig_seq = ltl.delay %sig, 0, 0 : i1
    %not_sig = ltl.not %sig_seq : !ltl.sequence
    verif.assert %not_sig : !ltl.property
    verif.assert %sig_seq {bmc.final} : !ltl.sequence
    verif.yield %sig : i1
  }
  func.return %bmc : i1
}

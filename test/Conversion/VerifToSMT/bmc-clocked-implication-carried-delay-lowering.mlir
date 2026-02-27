// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Regression: implication rewrite can leave a carried-state ltl.delay in the
// outlined bmc_circuit helper. Lower it to a bool derived from the current
// input instead of tripping strict temporal legalization.
// CHECK-LABEL: func.func @bmc_circuit(
// CHECK-NOT: ltl.delay
// CHECK: %[[ONE:.*]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK: %[[CARRY:.*]] = smt.eq %arg0, %[[ONE]] : !smt.bv<1>
// CHECK: return %arg0, %arg1, %arg2, %arg3, %arg0, %[[CARRY]], {{%.*}} : !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bool, !smt.bool
func.func @test_clocked_seq_implication() -> i1 {
  %bmc = verif.bmc bound 3 num_regs 0 initial_values []
  init {
    %false = arith.constant false
    verif.yield %false : i1
  }
  loop {
  ^bb0(%c0: i1):
    verif.yield %c0 : i1
  }
  circuit {
  ^bb0(%a: i1, %b: i1, %clk: i1, %en: i1):
    %d1 = ltl.delay %a, 1, 0 : i1
    %d2 = ltl.delay %b, 1, 0 : i1
    %p = ltl.implication %d1, %d2 : !ltl.sequence, !ltl.sequence
    verif.clocked_assert %p if %en, posedge %clk : !ltl.property
    verif.yield %a, %b, %clk, %en : i1, i1, i1, i1
  }
  func.return %bmc : i1
}

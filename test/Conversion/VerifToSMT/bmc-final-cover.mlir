// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @test_final_cover
// Final cover checks should be asserted directly (not negated).
// CHECK: smt.push 1
// CHECK: [[FINAL_EQ:%.+]] = smt.eq {{.*}}
// CHECK-NOT: smt.not [[FINAL_EQ]]
// CHECK: smt.assert [[FINAL_EQ]]
// CHECK: smt.check
// CHECK: smt.pop 1
func.func @test_final_cover() -> i1 {
  %bmc = verif.bmc bound 2 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1, %b: i1):
    %seq = ltl.delay %a, 0, 0 : i1
    verif.cover %seq : !ltl.sequence
    verif.cover %b {bmc.final} : i1
    verif.yield %a, %b : i1, i1
  }
  func.return %bmc : i1
}

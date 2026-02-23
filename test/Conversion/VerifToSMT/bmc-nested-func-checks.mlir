// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @test_nested_func_asserts
// CHECK: %[[CIRCUIT:.*]]:4 = func.call @bmc_circuit
// CHECK-SAME: -> (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>)
// CHECK: %[[CHECK0:.*]] = smt.eq %[[CIRCUIT]]#2, %c-1_bv1 : !smt.bv<1>
// CHECK: %[[NOT0:.*]] = smt.not %[[CHECK0]]
// CHECK: %[[CHECK1:.*]] = smt.eq %[[CIRCUIT]]#3, %c-1_bv1 : !smt.bv<1>
// CHECK: %[[NOT1:.*]] = smt.not %[[CHECK1]]
// CHECK: smt.or %[[NOT0]], %[[NOT1]]
func.func @test_nested_func_asserts() -> i1 {
  %bmc = verif.bmc bound 2 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1, %b: i1):
    func.call @assert_helper(%a) : (i1) -> ()
    func.call @assert_helper(%b) : (i1) -> ()
    verif.yield %a, %b : i1, i1
  }
  func.return %bmc : i1
}

func.func @assert_helper(%x: i1) {
  verif.assert %x : i1
  func.return
}

// CHECK-LABEL: func.func @bmc_circuit
// CHECK-NOT: func.call @assert_helper
// CHECK: return %arg0, %arg1, %arg0, %arg1 : !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>

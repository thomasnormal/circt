// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @test_nested_instance_asserts
// CHECK: %[[CIRCUIT:.*]]:4 = func.call @bmc_circuit
// CHECK-SAME: -> (!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>)
// CHECK: %[[CHECK0:.*]] = smt.eq %[[CIRCUIT]]#2, %c-1_bv1 : !smt.bv<1>
// CHECK: %[[NOT0:.*]] = smt.not %[[CHECK0]]
// CHECK: %[[CHECK1:.*]] = smt.eq %[[CIRCUIT]]#3, %c-1_bv1 : !smt.bv<1>
// CHECK: %[[NOT1:.*]] = smt.not %[[CHECK1]]
// CHECK: smt.or %[[NOT0]], %[[NOT1]]
func.func @test_nested_instance_asserts() -> i1 {
  %bmc = verif.bmc bound 2 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%a: i1, %b: i1):
    hw.instance "" @assert_pair(x: %a: i1, y: %b: i1) -> ()
    verif.yield %a, %b : i1, i1
  }
  func.return %bmc : i1
}

hw.module @assert_pair(in %x: i1, in %y: i1) {
  verif.assert %x : i1
  verif.assert %y : i1
}

// CHECK-LABEL: func.func @bmc_circuit
// CHECK-NOT: hw.instance
// CHECK: return %arg0, %arg1, %arg0, %arg1 : !smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<1>

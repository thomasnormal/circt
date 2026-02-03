// RUN: circt-opt %s --convert-verif-to-smt="assume-known-inputs=true" --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @test_bmc_known_inputs
// CHECK: [[IN:%.+]] = smt.declare_fun : !smt.bv<2>
// CHECK: [[UNK:%.+]] = smt.bv.extract [[IN]] from 0 : (!smt.bv<2>) -> !smt.bv<1>
// CHECK: [[KNOWN:%.+]] = smt.eq [[UNK]], %c0_bv1 : !smt.bv<1>
// CHECK: smt.assert [[KNOWN]]
func.func @test_bmc_known_inputs() -> i1 {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values []
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%sig: !hw.struct<value: i1, unknown: i1>):
    %val = hw.struct_extract %sig["value"] : !hw.struct<value: i1, unknown: i1>
    verif.assert %val : i1
    verif.yield %sig : !hw.struct<value: i1, unknown: i1>
  }
  func.return %bmc : i1
}

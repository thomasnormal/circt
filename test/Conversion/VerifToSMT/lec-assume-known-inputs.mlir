// RUN: circt-opt %s --convert-verif-to-smt="assume-known-inputs=true" --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

hw.type_scope @types {
  hw.typedecl @logic1_4s : !hw.struct<value: i1, unknown: i1>
}

// CHECK: smt.solver() : () -> i1 {
// CHECK: [[IN:%.+]] = smt.declare_fun : !smt.bv<2>
// CHECK: [[UNK:%.+]] = smt.bv.extract [[IN]] from 0 : (!smt.bv<2>) -> !smt.bv<1>
// CHECK: [[KNOWN:%.+]] = smt.eq [[UNK]], %c0_bv1 : !smt.bv<1>
// CHECK: smt.assert [[KNOWN]]
// CHECK: [[C1OUT:%.+]] = smt.declare_fun "c1_out0" : !smt.bv<2>
// CHECK: [[C2OUT:%.+]] = smt.declare_fun "c2_out0" : !smt.bv<2>
// CHECK: [[EQ1:%.+]] = smt.eq [[C1OUT]], [[IN]] : !smt.bv<2>
// CHECK: smt.assert [[EQ1]]
// CHECK: [[EQ2:%.+]] = smt.eq [[C2OUT]], [[IN]] : !smt.bv<2>
// CHECK: smt.assert [[EQ2]]
// CHECK: [[DIST:%.+]] = smt.distinct [[C1OUT]], [[C2OUT]] : !smt.bv<2>
// CHECK: smt.assert [[DIST]]
// CHECK: smt.check sat
// CHECK: }
func.func @test_lec_known_inputs() -> i1 {
  %0 = verif.lec : i1 first {
  ^bb0(%arg0: !hw.struct<value: i1, unknown: i1>):
    verif.yield %arg0 : !hw.struct<value: i1, unknown: i1>
  } second {
  ^bb0(%arg0: !hw.struct<value: i1, unknown: i1>):
    verif.yield %arg0 : !hw.struct<value: i1, unknown: i1>
  }
  return %0 : i1
}

// CHECK-LABEL: func @test_lec_known_inputs_alias
// CHECK: [[IN:%.+]] = smt.declare_fun : !smt.bv<2>
// CHECK: [[UNK:%.+]] = smt.bv.extract [[IN]] from 0 : (!smt.bv<2>) -> !smt.bv<1>
// CHECK: [[KNOWN:%.+]] = smt.eq [[UNK]], %c0_bv1 : !smt.bv<1>
// CHECK: smt.assert [[KNOWN]]
func.func @test_lec_known_inputs_alias() -> i1 {
  %0 = verif.lec : i1 first {
  ^bb0(%arg0: !hw.typealias<@types::@logic1_4s, !hw.struct<value: i1, unknown: i1>>):
    verif.yield %arg0 : !hw.typealias<@types::@logic1_4s, !hw.struct<value: i1, unknown: i1>>
  } second {
  ^bb0(%arg0: !hw.typealias<@types::@logic1_4s, !hw.struct<value: i1, unknown: i1>>):
    verif.yield %arg0 : !hw.typealias<@types::@logic1_4s, !hw.struct<value: i1, unknown: i1>>
  }
  return %0 : i1
}

// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s

module {
  // CHECK-LABEL: hw.module @disable_iff_nested
  hw.module @disable_iff_nested(in %clock : !seq.clock, in %disable1 : i1,
                                in %disable2 : i1, in %a : i1) {
    %eventually = ltl.eventually %a : i1
    %inner = ltl.or %disable2, %eventually {sva.disable_iff} : i1, !ltl.property
    %outer = ltl.or %disable1, %inner {sva.disable_iff} : i1, !ltl.property
    verif.assert %outer : !ltl.property
    // CHECK: %[[COMBINED:.+]] = comb.or %disable1, %disable2 : i1
    // CHECK: %[[SEEN:.+]] = seq.compreg {{.*}} %clock reset %[[COMBINED]]
    // CHECK: %[[INNER_FINAL:.+]] = comb.or {{.*}} %disable2, %[[SEEN]] : i1
    // CHECK: %[[OUTER_FINAL:.+]] = comb.or {{.*}} %disable1, %[[INNER_FINAL]] : i1
    // CHECK: verif.assert %[[OUTER_FINAL]] {bmc.final} : i1
    hw.output
  }
}

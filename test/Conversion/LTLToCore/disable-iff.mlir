// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s

module {
  // CHECK-LABEL: hw.module @disable_iff
  hw.module @disable_iff(in %clock : !seq.clock, in %disable : i1, in %a : i1) {
    %eventually = ltl.eventually %a : i1
    %disabled = ltl.or %disable, %eventually {sva.disable_iff} : i1, !ltl.property
    verif.assert %disabled : !ltl.property
    // CHECK: %[[SEEN:.+]] = seq.compreg {{.*}} %clock reset %disable
    // CHECK: %[[FINAL:.+]] = comb.or {{.*}} %disable, %[[SEEN]] : i1
    // CHECK: verif.assert %[[FINAL]] {bmc.final} : i1
    hw.output
  }
}

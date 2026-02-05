// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s

module {
  // CHECK-LABEL: hw.module @weak_eventually
  hw.module @weak_eventually(in %clock : !seq.clock, in %a : i1) {
    %prop = ltl.eventually %a {ltl.weak} : i1
    verif.assert %prop : !ltl.property
    // CHECK-NOT: ltl_eventually_seen
    // CHECK: verif.assert {{.*}} {bmc.final}
    hw.output
  }
}

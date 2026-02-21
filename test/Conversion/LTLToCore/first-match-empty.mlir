// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s

module {
  hw.module @first_match_empty(in %clk : i1, in %a : i1) {
    %empty = ltl.repeat %a, 0, 0 : i1
    %fm = ltl.first_match %empty : !ltl.sequence
    verif.clocked_assert %fm, posedge %clk : !ltl.sequence
    hw.output
  }
}

// CHECK-LABEL: hw.module @first_match_empty
// CHECK: verif.assert %true{{.*}} {bmc.clock = "clk", bmc.clock_edge = #ltl<clock_edge posedge>} : i1

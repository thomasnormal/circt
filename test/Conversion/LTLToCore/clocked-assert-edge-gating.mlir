// RUN: circt-opt %s --lower-clocked-assert-like --lower-ltl-to-core | FileCheck %s

module {
  hw.module @clocked_assert_negedge(in %clk : i1, in %a : i1) {
    verif.clocked_assert %a, negedge %clk : i1
    hw.output
  }

  hw.module @clocked_cover_edge(in %clk : i1, in %a : i1) {
    verif.clocked_cover %a, edge %clk : i1
    hw.output
  }
}

// CHECK: hw.module @clocked_assert_negedge
// CHECK: comb.xor %clk
// CHECK: seq.to_clock
// CHECK: seq.compreg sym @ltl_state
// CHECK: verif.assert
// CHECK: hw.module @clocked_cover_edge
// CHECK: seq.to_clock
// CHECK: seq.compreg sym @ltl_state
// CHECK: verif.cover

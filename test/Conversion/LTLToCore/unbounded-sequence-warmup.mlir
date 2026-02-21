// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s

module {
  hw.module @unbounded_sequence_warmup(in %clk: i1, in %a: i1) {
    // Sequence minimum length is 2 but max length is unbounded.
    // Assertion lowering should still apply one-cycle startup warmup.
    %rep = ltl.non_consecutive_repeat %a, 2 : i1
    verif.clocked_assert %rep, posedge %clk : !ltl.sequence
    hw.output
  }
}

// CHECK-LABEL: hw.module @unbounded_sequence_warmup
// CHECK: %[[PAST:.*]] = seq.compreg sym @ltl_past
// CHECK: %[[NW:.*]] = comb.xor %[[PAST]], %{{.*}} : i1
// CHECK: %[[GATED:.*]] = comb.or bin %[[NW]], %{{.*}} : i1
// CHECK: verif.assert %[[GATED]] {bmc.clock = "clk", bmc.clock_edge = #ltl<clock_edge posedge>} : i1

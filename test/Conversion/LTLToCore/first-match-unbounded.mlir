// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s

module {
  hw.module @unbounded_first_match(in %clk: i1, in %a: i1) {
    // Unbounded non-consecutive repeat under first_match should lower without
    // emitting a hard error.
    // We also verify structural first-match behavior in the lowered form:
    // once a match happens, next-state updates are masked off by `!match`.
    %rep = ltl.non_consecutive_repeat %a, 1 : i1
    %fm = ltl.first_match %rep : !ltl.sequence
    verif.clocked_assert %fm, posedge %clk : !ltl.sequence
    hw.output
  }
}

// CHECK-LABEL: hw.module @unbounded_first_match
// CHECK: %[[NOT_MATCH:.*]] = comb.xor %[[MATCH:.*]], %{{.*}} : i1
// CHECK: comb.and bin %{{.*}}, %[[NOT_MATCH]] : i1
// CHECK: verif.assert %[[MATCH]] {bmc.clock = "clk", bmc.clock_edge = #ltl<clock_edge posedge>} : i1

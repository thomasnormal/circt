// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s

// Test that clocked assertions with implication properties are properly lowered.
// The implication tracking requires state registers to track when the antecedent
// has been seen.

module {
  hw.module @test(in %clock : !seq.clock, in %a : i1) {
    %clk = seq.from_clock %clock
    %seq = ltl.delay %a, 0, 0 : i1
    %prop = ltl.implication %a, %seq : i1, !ltl.sequence
    verif.clocked_assert %prop, posedge %clk : !ltl.property
    hw.output
  }
}

// CHECK: hw.module @test
// CHECK: seq.compreg sym @ltl_state
// CHECK: seq.compreg sym @ltl_implication_seen
// CHECK: verif.assert {{.*}} {bmc.clock = "clock"

// RUN: circt-opt %s --lower-clocked-assert-like --lower-ltl-to-core | FileCheck %s

// Test that clocked assertions are properly lowered with clock gating.
// A simple i1 clocked assertion should be converted to NFA-based state tracking.

module {
  hw.module @clocked_property_gating(in %clk : i1, in %a : i1) {
    verif.clocked_assert %a, posedge %clk : i1
    hw.output
  }
}

// CHECK: hw.module @clocked_property_gating
// CHECK: seq.to_clock
// CHECK: seq.compreg sym @ltl_state
// CHECK: verif.assert {{.*}} {bmc.final} : i1
// CHECK: verif.assert

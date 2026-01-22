// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s
// Test that negating a sequence produces a safety property with no liveness requirement.
// The finalCheck for a negated sequence should be true, not negated.

module {
  // CHECK-LABEL: hw.module @not_sequence
  hw.module @not_sequence(in %clock : !seq.clock, in %a : i1, in %b : i1) {
    // Create a simple sequence: a followed by b
    %delay_a = ltl.delay %a, 0, 0 : i1
    %delay_b = ltl.delay %b, 1, 0 : i1
    %concat = ltl.concat %delay_a, %delay_b : !ltl.sequence, !ltl.sequence

    // Negate the sequence: not (a ##1 b)
    %not_seq = ltl.not %concat : !ltl.sequence

    // Assert the negated sequence
    verif.assert %not_seq : !ltl.property

    // The finalCheck assertion should have bmc.final attribute and be true (no liveness)
    // CHECK: verif.assert %true{{.*}} {bmc.final} : i1
    // CHECK: verif.assert {{.*}} : i1
    hw.output
  }

  // CHECK-LABEL: hw.module @not_simple_i1
  hw.module @not_simple_i1(in %clock : !seq.clock, in %a : i1) {
    // Negate a simple i1 (single-cycle sequence)
    %not_a = ltl.not %a : i1

    // Assert the negated i1
    verif.assert %not_a : !ltl.property

    // The finalCheck assertion should have bmc.final attribute and be true (no liveness)
    // CHECK: verif.assert %true{{.*}} {bmc.final} : i1
    // CHECK: verif.assert {{.*}} : i1
    hw.output
  }
}

// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s

module {
  // CHECK-LABEL: hw.module @seq_bounds_or_and
  hw.module @seq_bounds_or_and(in %clock : !seq.clock, in %a : i1, in %b : i1) {
    %clk = seq.from_clock %clock
    %a1 = ltl.delay %a, 1, 0 : i1
    %b1 = ltl.delay %b, 1, 0 : i1
    %seq_or = ltl.or %a1, %b1 : !ltl.sequence, !ltl.sequence
    %seq_and = ltl.and %a1, %b1 : !ltl.sequence, !ltl.sequence
    %not_or = ltl.not %seq_or : !ltl.sequence
    %not_and = ltl.not %seq_and : !ltl.sequence
    %clocked_or = ltl.clock %not_or, posedge %clk : !ltl.property
    %clocked_and = ltl.clock %not_and, posedge %clk : !ltl.property
    verif.assert %clocked_or : !ltl.property
    verif.assert %clocked_and : !ltl.property
    // CHECK: seq.compreg sym @ltl_past
    hw.output
  }
}

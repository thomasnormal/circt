// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s

// Test that a clocked sequence assertion is properly lowered.
// The sequence should be converted to NFA-based state registers,
// with warmup logic to avoid false failures during sequence startup.

hw.module @clocked_sequence_assert(in %clk : i1, in %a : i1, in %b : i1) {
  %sa = ltl.delay %a, 0, 0 : i1
  %sb = ltl.delay %b, 0, 0 : i1
  %seq = ltl.concat %sa, %sb : !ltl.sequence, !ltl.sequence
  %clocked = ltl.clock %seq, posedge %clk : !ltl.sequence
  // CHECK: seq.compreg sym @ltl_state
  // CHECK: seq.compreg sym @ltl_past
  // CHECK: verif.assert %true{{.*}} {bmc.final} : i1
  // CHECK: verif.assert %{{.*}} : i1
  verif.assert %clocked : !ltl.sequence
  hw.output
}

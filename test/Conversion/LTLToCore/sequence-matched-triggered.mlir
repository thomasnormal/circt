// RUN: circt-opt --lower-ltl-to-core %s | FileCheck %s

hw.module @sequence_match_trigger(in %clk: !seq.clock, in %a: i1) {
  %seq = ltl.delay %a, 0, 0 : i1
  %matched = ltl.matched %seq : !ltl.sequence -> i1
  %triggered = ltl.triggered %seq : !ltl.sequence -> i1
  %both = comb.and %matched, %triggered : i1
  verif.assert %both : i1
  hw.output
}

// CHECK: ltl_state
// CHECK: ltl_past
// CHECK-NOT: ltl.matched
// CHECK-NOT: ltl.triggered

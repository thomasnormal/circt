// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// Test case for cross-module event trigger
// The initial block in @inner references a port that comes from outside,
// so it should use llhd.process (not seq.initial which has IsolatedFromAbove)
//
// Also tests that event_trigger passes the actual event ref address to
// __moore_event_trigger, not a temporary copy.

// CHECK-LABEL: hw.module @inner
// CHECK-SAME: in %top_e : !llhd.ref<i1>
// CHECK: llhd.process
// CHECK-NOT: seq.initial
// The event trigger takes the ref directly and toggles the underlying signal
// CHECK: llhd.prb %top_e
// CHECK: llhd.drv %top_e

moore.module @inner(in %top_e : !moore.ref<event>) {
  moore.procedure initial {
    moore.event_trigger %top_e : <event>
    moore.return
  }
  moore.output
}

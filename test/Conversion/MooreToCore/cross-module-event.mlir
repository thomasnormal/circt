// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// Test case for cross-module event trigger
// The initial block in @inner references a port that comes from outside,
// so it should use llhd.process (not seq.initial which has IsolatedFromAbove)

// CHECK-LABEL: hw.module @inner
// CHECK: llhd.process
// CHECK-NOT: seq.initial

moore.module @inner(in %top_e : !moore.ref<event>) {
  moore.procedure initial {
    %0 = moore.read %top_e : <event>
    moore.event_trigger %0 : event
    moore.return
  }
  moore.output
}

// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test delayed non-blocking event trigger lowering.
// `->> #d e` should be accepted and lowered as nonblocking work.

// CHECK-LABEL: moore.module @NonblockingDelayedEventTrigger
// CHECK: moore.procedure initial
// CHECK: moore.fork join_none
// CHECK: moore.wait_delay
// CHECK: moore.event_trigger
module NonblockingDelayedEventTrigger;
  event e;
  initial ->> #1 e;
endmodule

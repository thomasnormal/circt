// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test hierarchical event references in procedural blocks.
// This tests that the hierarchical name collector properly traverses
// procedural blocks to find hierarchical event triggers.

// Test 1: Downward hierarchical reference (parent referencing child's event)
// CHECK-LABEL: moore.module @Top
module Top;
  event e;

  Inner inner();

  // CHECK: moore.procedure initial
  initial begin
    // Wait for the event from inner module - uses hierarchical reference
    // CHECK: moore.wait_event
    // CHECK: moore.read %inner.done
    // CHECK: moore.detect_event any
    @(inner.done);
  end
endmodule

// CHECK-LABEL: moore.module private @Inner
// CHECK-SAME: out done : !moore.ref<event>
module Inner;
  event done;

  // CHECK: moore.procedure initial
  initial begin
    // Trigger event - blocking trigger
    // CHECK: moore.event_trigger
    -> done;
    #10;
    // Nonblocking trigger
    // CHECK: moore.event_trigger
    ->> done;
  end
endmodule

// Test 2: Upward hierarchical reference (child triggering parent's event)
// This is the key use case where inner module triggers an event in the parent.
// CHECK-LABEL: moore.module @TopWithUpward
module TopWithUpward;
  event sync_event;

  // CHECK: moore.instance "inner2"
  Inner2 inner2();

  initial begin
    // Wait for event triggered by inner module
    // CHECK: moore.wait_event
    @(sync_event);
    $display("Event received from inner module");
  end
endmodule

// CHECK-LABEL: moore.module private @Inner2
// The sync_event should be threaded through as an input port
// CHECK-SAME: in {{.*}}sync_event : !moore.ref<event>
module Inner2;
  // CHECK: moore.procedure initial
  initial begin
    #5;
    // Trigger parent's event using hierarchical reference
    // This requires the hierarchical name collector to traverse the procedural block
    // CHECK: moore.event_trigger
    -> TopWithUpward.sync_event;
    #10;
    // Nonblocking trigger to parent's event
    // CHECK: moore.event_trigger
    ->> TopWithUpward.sync_event;
  end
endmodule

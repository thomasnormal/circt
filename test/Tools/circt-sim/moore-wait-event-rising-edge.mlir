// RUN: circt-sim %s --max-time=5000000 2>&1 | FileCheck %s

// Test that moore.wait_event with !moore.event type uses RISING EDGE detection.
// This is critical for UVM-style events where @(event) should wait for the
// event to be TRIGGERED (->event), which is a 0->1 transition.
//
// The process should only wake on 0->1 transitions, NOT on 1->0 transitions.
// This is essential for UVM wait_for_objection semantics where:
// - If no objection has been raised yet (value=0), wait for NEXT trigger
// - Don't wake on objection drop (1->0 transition)

// CHECK: Starting simulation
// CHECK: At 1ns: Setting event to TRUE
// CHECK: Trigger #1 detected!
// CHECK: At 2ns: Setting event to FALSE (should not trigger waiter)
// CHECK: At 3ns: Setting event to TRUE
// CHECK: Trigger #2 detected! Test PASSED

hw.module @RisingEdgeTest() {
  %true = hw.constant true
  %false = hw.constant false
  %zero = hw.constant 0 : i32
  %one_i64 = hw.constant 1 : i64
  %one = hw.constant 1 : i32
  %two = hw.constant 2 : i32
  %delay_1ns = llhd.constant_time <1ns, 0d, 0e>

  // Format strings for output
  %fmt_start = sim.fmt.literal "Starting simulation\0A"
  %fmt_trigger1 = sim.fmt.literal "Trigger #1 detected!\0A"
  %fmt_trigger2 = sim.fmt.literal "Trigger #2 detected! Test PASSED\0A"
  %fmt_set_true1 = sim.fmt.literal "At 1ns: Setting event to TRUE\0A"
  %fmt_set_false = sim.fmt.literal "At 2ns: Setting event to FALSE (should not trigger waiter)\0A"
  %fmt_set_true2 = sim.fmt.literal "At 3ns: Setting event to TRUE\0A"

  // Dummy signal
  %tick = llhd.sig %false : i1

  // Allocate memory for our "UVM event" boolean field
  %event_ptr = llvm.alloca %one_i64 x i1 : (i64) -> !llvm.ptr

  // Initialize the event to false (not triggered)
  llvm.store %false, %event_ptr : i1, !llvm.ptr

  // Allocate counter for triggers
  %counter_ptr = llvm.alloca %one_i64 x i32 : (i64) -> !llvm.ptr
  llvm.store %zero, %counter_ptr : i32, !llvm.ptr

  // Process 1: Wait for the memory event repeatedly (using rising edge detection)
  llhd.process {
    cf.br ^wait_event
  ^wait_event:
    // Wait for the memory-based event (should use rising edge detection for !moore.event)
    moore.wait_event {
      %event_val = llvm.load %event_ptr : !llvm.ptr -> i1
      // Convert to moore.event type - this triggers rising edge detection
      %event = builtin.unrealized_conversion_cast %event_val : i1 to !moore.event
      moore.detect_event any %event : event
    }
    // Increment counter
    %count = llvm.load %counter_ptr : !llvm.ptr -> i32
    %new_count = comb.add %count, %one : i32
    llvm.store %new_count, %counter_ptr : i32, !llvm.ptr
    // Check if this is trigger #1 or #2
    %is_first = comb.icmp bin eq %new_count, %one : i32
    cf.cond_br %is_first, ^first_trigger, ^second_trigger
  ^first_trigger:
    sim.proc.print %fmt_trigger1
    // Go back to waiting - should NOT wake on 1->0 transition at t=2ns
    cf.br ^wait_event
  ^second_trigger:
    sim.proc.print %fmt_trigger2
    llhd.halt
  }

  // Process 2: Sequence of writes to test edge detection
  llhd.process {
    // At t=1ns: Set to true (should trigger waiter)
    llhd.wait delay %delay_1ns, ^set_true1
  ^set_true1:
    sim.proc.print %fmt_set_true1
    llvm.store %true, %event_ptr : i1, !llvm.ptr
    // At t=2ns: Set to false (should NOT trigger waiter - falling edge)
    llhd.wait delay %delay_1ns, ^set_false
  ^set_false:
    sim.proc.print %fmt_set_false
    llvm.store %false, %event_ptr : i1, !llvm.ptr
    // At t=3ns: Set to true (should trigger waiter again)
    llhd.wait delay %delay_1ns, ^set_true2
  ^set_true2:
    sim.proc.print %fmt_set_true2
    llvm.store %true, %event_ptr : i1, !llvm.ptr
    llhd.halt
  }

  // Print startup message
  llhd.process {
    sim.proc.print %fmt_start
    llhd.halt
  }

  hw.output
}

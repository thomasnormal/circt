// RUN: circt-sim %s --max-time=10000000 2>&1 | FileCheck %s

// Test that moore.wait_event can wait on memory-based boolean events.
// This simulates UVM events stored as boolean fields in class instances
// where no hardware signal is available to wait on.
//
// The pattern being tested:
//   %ptr = llvm.getelementptr %obj[0, N] : (!llvm.ptr) -> !llvm.ptr
//   %val = llvm.load %ptr : !llvm.ptr -> i1
//   %evt = builtin.unrealized_conversion_cast %val : i1 to !moore.event
//   moore.detect_event any %evt : event
//
// The memory event polling mechanism should detect when the boolean
// value at the memory location changes and wake the waiting process.

// CHECK: Starting simulation
// CHECK: Triggering event by writing to memory.
// CHECK: Event triggered! Value changed.
// CHECK: Simulation completed at time 1000000 fs

hw.module @MemoryEventTest() {
  %true = hw.constant true
  %false = hw.constant false
  %one = hw.constant 1 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %delay_1ns = llhd.constant_time <1ns, 0d, 0e>

  // Format strings for output
  %fmt_start = sim.fmt.literal "Starting simulation\0A"
  %fmt_event = sim.fmt.literal "Event triggered! Value changed.\0A"
  %fmt_trigger = sim.fmt.literal "Triggering event by writing to memory.\0A"
  %fmt_done = sim.fmt.literal "Simulation completed\0A"

  // Dummy signal to drive the simulation forward
  %tick = llhd.sig %false : i1

  // Allocate memory for our "UVM event" boolean field
  // This simulates a class instance with a boolean event field
  %event_ptr = llvm.alloca %one x i1 : (i64) -> !llvm.ptr

  // Initialize the event to false (not triggered)
  llvm.store %false, %event_ptr : i1, !llvm.ptr

  // Process 1: Wait for the memory event to be triggered
  // This simulates a UVM objection.wait_for(dropped) call
  llhd.process {
    cf.br ^wait_event
  ^wait_event:
    // Wait for the memory-based event
    moore.wait_event {
      // Load the boolean from memory
      %event_val = llvm.load %event_ptr : !llvm.ptr -> i1
      // Convert to moore.event type
      %event = builtin.unrealized_conversion_cast %event_val : i1 to !moore.event
      // Detect any change on this event
      moore.detect_event any %event : event
    }
    // When we get here, the event was triggered
    sim.proc.print %fmt_event
    llhd.halt
  }

  // Process 2: After some delay, trigger the event by writing to memory
  // This simulates another UVM component calling objection.drop()
  llhd.process {
    // Initial wait to let Process 1 start waiting
    llhd.wait delay %delay_1ns, ^trigger
  ^trigger:
    // Trigger the event by writing true to the memory location
    llvm.store %true, %event_ptr : i1, !llvm.ptr
    sim.proc.print %fmt_trigger
    llhd.halt
  }

  // Print startup message
  llhd.process {
    sim.proc.print %fmt_start
    llhd.halt
  }

  hw.output
}

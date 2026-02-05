// RUN: circt-sim %s --max-time=10000000 2>&1 | FileCheck %s

// Test that moore.wait_event can wait on events stored in class member variables.
// This simulates UVM objection events stored in heap-allocated class instances
// where the event field is accessed through a pointer chain:
//   1. Load class instance pointer from heap (malloc)
//   2. Use GEP to compute the address of the event member field
//   3. Load the event value from the member field
//   4. Wait for the event to be triggered
//
// The pattern being tested:
//   %class_ptr = llvm.load %slot : !llvm.ptr -> !llvm.ptr
//   %event_ptr = llvm.getelementptr %class_ptr[0, event_field_idx] : ...
//   %val = llvm.load %event_ptr : !llvm.ptr -> i1
//   %evt = builtin.unrealized_conversion_cast %val : i1 to !moore.event
//   moore.detect_event any %evt : event
//
// This requires on-demand evaluation of GEP and load operations during
// memory event tracing in moore.wait_event.

// CHECK: Starting simulation
// CHECK: Triggering event in class member.
// CHECK: Event triggered! Class member event received.
// CHECK: Simulation completed at time 1000000 fs

!class_type = !llvm.struct<"UvmObjection", (i1)>

hw.module @WaitEventClassMember() {
  %true = hw.constant true
  %false = hw.constant false
  %one = hw.constant 1 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %delay_1ns = llhd.constant_time <1ns, 0d, 0e>

  // Format strings for output
  %fmt_start = sim.fmt.literal "Starting simulation\0A"
  %fmt_event = sim.fmt.literal "Event triggered! Class member event received.\0A"
  %fmt_trigger = sim.fmt.literal "Triggering event in class member.\0A"
  %fmt_done = sim.fmt.literal "Simulation completed\0A"

  // Dummy signal to drive the simulation forward
  %tick = llhd.sig %false : i1

  // Simulate a class instance stored in heap memory:
  // 1. Allocate a "slot" pointer (like a class reference variable)
  // 2. Allocate the actual class instance (like malloc for a class)
  // 3. Store the class pointer in the slot

  // The slot that holds the pointer to the class instance
  %slot = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr

  // The class instance (struct with one i1 field for the event)
  %class_size = hw.constant 1 : i64
  %class_instance = llvm.alloca %class_size x !class_type : (i64) -> !llvm.ptr

  // Store the class pointer in the slot (simulates: MyClass obj = new();)
  llvm.store %class_instance, %slot : !llvm.ptr, !llvm.ptr

  // Initialize the event field in the class to false (event not triggered)
  %zero_idx = llvm.mlir.constant(0 : i32) : i32
  %event_field_ptr = llvm.getelementptr inbounds %class_instance[%zero_idx, 0]
      : (!llvm.ptr, i32) -> !llvm.ptr, !class_type
  llvm.store %false, %event_field_ptr : i1, !llvm.ptr

  // Process 1: Wait for the memory event stored in the class member
  // This simulates: wait(objection.dropped_event.triggered);
  llhd.process {
    cf.br ^wait_event
  ^wait_event:
    // Wait for the memory-based event in the class member
    moore.wait_event {
      // Load the class pointer from the slot (simulates: accessing obj)
      %loaded_class_ptr = llvm.load %slot : !llvm.ptr -> !llvm.ptr
      // Compute address of event field (simulates: obj.dropped_event)
      %loaded_event_ptr = llvm.getelementptr inbounds %loaded_class_ptr[%zero_idx, 0]
          : (!llvm.ptr, i32) -> !llvm.ptr, !class_type
      // Load the event value
      %event_val = llvm.load %loaded_event_ptr : !llvm.ptr -> i1
      // Convert to moore.event type
      %event = builtin.unrealized_conversion_cast %event_val : i1 to !moore.event
      // Detect any change on this event
      moore.detect_event any %event : event
    }
    // When we get here, the event was triggered
    sim.proc.print %fmt_event
    llhd.halt
  }

  // Process 2: After some delay, trigger the event by writing to the class member
  // This simulates: objection.drop() which sets dropped_event = true
  llhd.process {
    // Initial wait to let Process 1 start waiting
    llhd.wait delay %delay_1ns, ^trigger
  ^trigger:
    // Trigger the event by writing true to the class member
    // Load class pointer and compute event field address
    %trigger_class_ptr = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %trigger_event_ptr = llvm.getelementptr inbounds %trigger_class_ptr[%zero_idx, 0]
        : (!llvm.ptr, i32) -> !llvm.ptr, !class_type
    // Write true to trigger the event
    llvm.store %true, %trigger_event_ptr : i1, !llvm.ptr
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

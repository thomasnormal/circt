// RUN: circt-sim %s 2>&1 | FileCheck %s
// Test for __moore_event_trigger and __moore_event_triggered runtime functions.

// CHECK: Event trigger test
// CHECK: Event was triggered!

llvm.func @__moore_event_trigger(!llvm.ptr)
llvm.func @__moore_event_triggered(!llvm.ptr) -> i1

hw.module @EventTriggerTest() {
  %lit_test = sim.fmt.literal "Event trigger test\0A"
  %lit_triggered = sim.fmt.literal "Event was triggered!\0A"
  %lit_not_triggered = sim.fmt.literal "Event was NOT triggered\0A"
  %one = llvm.mlir.constant(1 : i64) : i64
  %false = hw.constant false

  llhd.process {
    sim.proc.print %lit_test

    // Create a single alloca that represents the event storage
    %event_storage = llvm.alloca %one x i1 : (i64) -> !llvm.ptr
    llvm.store %false, %event_storage : i1, !llvm.ptr

    // Trigger the event
    llvm.call @__moore_event_trigger(%event_storage) : (!llvm.ptr) -> ()

    // Check if triggered (using same storage)
    %triggered = llvm.call @__moore_event_triggered(%event_storage) : (!llvm.ptr) -> i1

    // Print result
    %msg = arith.select %triggered, %lit_triggered, %lit_not_triggered : !sim.fstring
    sim.proc.print %msg

    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}

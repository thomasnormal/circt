// RUN: circt-sim %s --max-time 100000000 2>&1 | FileCheck %s
//
// Test that phase hopper objection interceptors work correctly.
// This specifically tests:
// 1. get_objection/raise/drop for uvm_phase_hopper:: (not just uvm_phase::)
// 2. wait_for blocks until objections are raised AND dropped (wasEverRaised)
// 3. Without wasEverRaised tracking, wait_for would return at time 0
//
// Process A raises an objection at time 0, then drops it at 5ns.
// Process B calls wait_for, which must block until after the drop.
//
// CHECK: raised
// CHECK: dropped
// CHECK: wait_for done
// CHECK: [circt-sim] Simulation completed

module {
  // A global to act as the shared "phase" object address.
  llvm.mlir.global internal @hopper_phase(0 : i64) : i64

  // Phase hopper functions with dummy bodies. The interpreter intercepts
  // these by name before executing the body, so the body is never reached.
  func.func @"uvm_pkg::uvm_phase_hopper::get_objection"(%phase: !llvm.ptr) -> !llvm.ptr {
    %null = llvm.mlir.zero : !llvm.ptr
    return %null : !llvm.ptr
  }
  func.func @"uvm_pkg::uvm_phase_hopper::raise_objection"(%phase: !llvm.ptr) {
    return
  }
  func.func @"uvm_pkg::uvm_phase_hopper::drop_objection"(%phase: !llvm.ptr) {
    return
  }
  func.func @"uvm_pkg::uvm_objection::wait_for"(%obj: !llvm.ptr) {
    return
  }

  hw.module @top() {
    %t5ns = llhd.constant_time <5ns, 0d, 0e>
    %fmt_raised = sim.fmt.literal "raised\0A"
    %fmt_dropped = sim.fmt.literal "dropped\0A"
    %fmt_done = sim.fmt.literal "wait_for done\0A"

    // Process A: raise objection, delay 5ns, drop objection
    llhd.process {
      cf.br ^start
    ^start:
      %phase = llvm.mlir.addressof @hopper_phase : !llvm.ptr
      %obj = func.call @"uvm_pkg::uvm_phase_hopper::get_objection"(%phase) : (!llvm.ptr) -> !llvm.ptr
      func.call @"uvm_pkg::uvm_phase_hopper::raise_objection"(%phase) : (!llvm.ptr) -> ()
      sim.proc.print %fmt_raised
      llhd.wait delay %t5ns, ^drop
    ^drop:
      %phase2 = llvm.mlir.addressof @hopper_phase : !llvm.ptr
      func.call @"uvm_pkg::uvm_phase_hopper::drop_objection"(%phase2) : (!llvm.ptr) -> ()
      sim.proc.print %fmt_dropped
      llhd.halt
    }

    // Process B: wait for objection to be raised AND dropped
    llhd.process {
      cf.br ^start
    ^start:
      %phase = llvm.mlir.addressof @hopper_phase : !llvm.ptr
      %obj = func.call @"uvm_pkg::uvm_phase_hopper::get_objection"(%phase) : (!llvm.ptr) -> !llvm.ptr
      func.call @"uvm_pkg::uvm_objection::wait_for"(%obj) : (!llvm.ptr) -> ()
      sim.proc.print %fmt_done
      llhd.halt
    }

    hw.output
  }
}

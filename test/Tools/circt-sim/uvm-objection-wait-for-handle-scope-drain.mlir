// RUN: circt-sim %s --max-time 200000000 2>&1 | FileCheck %s
//
// Regression for uvm_objection::wait_for interception:
// 1) wait_for state must be scoped by objection handle, not only process.
//    After waiting on one handle, a subsequent wait on another handle must
//    not reuse "wasEverRaised" state.
// 2) set_drain_time must be honored in simulation time.
//
// Sequence:
// - Objection A: raised at t=0, dropped at t=1ns.
// - Objection B: raised at t=20ns, dropped at t=30ns, drain_time=10ns.
// - Main process waits on A, then waits on B.
// Expected completion time is t=40ns = 40_000_000 fs.
//
// CHECK: raised A
// CHECK: drop A
// CHECK: wait A done
// CHECK: raise B
// CHECK: drop B
// CHECK: wait B done
// CHECK: [circt-sim] Simulation completed at time 40000000 fs

module {
  llvm.mlir.global internal @phase_a(0 : i64) : i64
  llvm.mlir.global internal @phase_b(0 : i64) : i64

  // Intercepted by name; bodies are never executed.
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
  func.func @"uvm_pkg::uvm_objection::wait_for"(%obj: !llvm.ptr, %objt_event: i32,
                                                 %source_obj: !llvm.ptr) {
    return
  }
  func.func @"uvm_pkg::uvm_objection::set_drain_time"(%obj: !llvm.ptr,
                                                       %source_obj: !llvm.ptr,
                                                       %drain_time: i64) {
    return
  }

  hw.module @top() {
    %line_raised_a = sim.fmt.literal "raised A\0A"
    %line_drop_a = sim.fmt.literal "drop A\0A"
    %line_wait_a = sim.fmt.literal "wait A done\0A"
    %line_raise_b = sim.fmt.literal "raise B\0A"
    %line_drop_b = sim.fmt.literal "drop B\0A"
    %line_wait_b = sim.fmt.literal "wait B done\0A"

    llhd.process {
      %null = llvm.mlir.zero : !llvm.ptr
      %c4_i32 = hw.constant 4 : i32
      %drain_10ns_fs = hw.constant 10000000 : i64
      %phase_a_addr = llvm.mlir.addressof @phase_a : !llvm.ptr
      %phase_b_addr = llvm.mlir.addressof @phase_b : !llvm.ptr

      %obj_a = func.call @"uvm_pkg::uvm_phase_hopper::get_objection"(%phase_a_addr) :
          (!llvm.ptr) -> !llvm.ptr
      %obj_b = func.call @"uvm_pkg::uvm_phase_hopper::get_objection"(%phase_b_addr) :
          (!llvm.ptr) -> !llvm.ptr

      // Raise A immediately.
      func.call @"uvm_pkg::uvm_phase_hopper::raise_objection"(%phase_a_addr) : (!llvm.ptr) -> ()
      sim.proc.print %line_raised_a

      // Configure drain time on B before waiting on it.
      func.call @"uvm_pkg::uvm_objection::set_drain_time"(%obj_b, %null,
                                                           %drain_10ns_fs) :
          (!llvm.ptr, !llvm.ptr, i64) -> ()

      // Wait for A to drop, then B to raise+drop+drain.
      func.call @"uvm_pkg::uvm_objection::wait_for"(%obj_a, %c4_i32, %null) :
          (!llvm.ptr, i32, !llvm.ptr) -> ()
      sim.proc.print %line_wait_a

      func.call @"uvm_pkg::uvm_objection::wait_for"(%obj_b, %c4_i32, %null) :
          (!llvm.ptr, i32, !llvm.ptr) -> ()
      sim.proc.print %line_wait_b

      sim.terminate success, quiet
      llhd.halt
    }

    // Drop A at t=1ns.
    llhd.process {
      %drop_a_fs = hw.constant 1000000 : i64
      %phase_a_addr = llvm.mlir.addressof @phase_a : !llvm.ptr

      %delay = llhd.int_to_time %drop_a_fs
      llhd.wait delay %delay, ^bb1
    ^bb1:
      func.call @"uvm_pkg::uvm_phase_hopper::drop_objection"(%phase_a_addr) : (!llvm.ptr) -> ()
      sim.proc.print %line_drop_a
      llhd.halt
    }

    // Raise B at t=20ns, drop B at t=30ns.
    llhd.process {
      %raise_b_fs = hw.constant 20000000 : i64
      %drop_b_fs = hw.constant 10000000 : i64
      %phase_b_addr = llvm.mlir.addressof @phase_b : !llvm.ptr

      %raise_delay = llhd.int_to_time %raise_b_fs
      llhd.wait delay %raise_delay, ^bb1
    ^bb1:
      func.call @"uvm_pkg::uvm_phase_hopper::raise_objection"(%phase_b_addr) : (!llvm.ptr) -> ()
      sim.proc.print %line_raise_b

      %drop_delay = llhd.int_to_time %drop_b_fs
      llhd.wait delay %drop_delay, ^bb2
    ^bb2:
      func.call @"uvm_pkg::uvm_phase_hopper::drop_objection"(%phase_b_addr) : (!llvm.ptr) -> ()
      sim.proc.print %line_drop_b
      llhd.halt
    }

    hw.output
  }
}

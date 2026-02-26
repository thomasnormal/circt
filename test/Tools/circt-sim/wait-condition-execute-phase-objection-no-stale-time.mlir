// RUN: circt-sim %s --top=top --sim-stats | FileCheck %s

// Regression: execute_phase wait(condition) can wake early via objection drop.
// A stale fallback poll must not keep the event queue alive and advance final
// simulation time far past the wakeup point.
//
// CHECK: done
// CHECK: [circt-sim] Simulation completed at time 5 fs

module {
  llvm.func @__moore_wait_condition(i32)
  llvm.func @__moore_delay(i64)

  llvm.mlir.global internal @flag(0 : i32) : i32

  func.func @"uvm_pkg::uvm_phase::raise_objection"(%phase: i64, %obj: !llvm.ptr,
                                                   %desc: !llvm.ptr, %count: i32) {
    return
  }
  func.func @"uvm_pkg::uvm_phase::drop_objection"(%phase: i64, %obj: !llvm.ptr,
                                                  %desc: !llvm.ptr, %count: i32) {
    return
  }

  llvm.func @"uvm_pkg::uvm_phase_hopper::execute_phase"(%self: i64,
                                                         %phase: i64) {
    %flagAddr = llvm.mlir.addressof @flag : !llvm.ptr
    %cond = llvm.load %flagAddr : !llvm.ptr -> i32
    llvm.call @__moore_wait_condition(%cond) : (i32) -> ()
    llvm.return
  }

  hw.module @top() {
    %done = sim.fmt.literal "done\0A"

    llhd.process {
      %self = llvm.mlir.constant(1 : i64) : i64
      %phase = llvm.mlir.constant(4096 : i64) : i64
      %null = llvm.mlir.zero : !llvm.ptr
      %one_i32 = hw.constant 1 : i32
      func.call @"uvm_pkg::uvm_phase::raise_objection"(%phase, %null, %null, %one_i32)
        : (i64, !llvm.ptr, !llvm.ptr, i32) -> ()
      llvm.call @"uvm_pkg::uvm_phase_hopper::execute_phase"(%self, %phase)
        : (i64, i64) -> ()
      sim.proc.print %done
      llhd.halt
    }

    llhd.process {
      %phase = llvm.mlir.constant(4096 : i64) : i64
      %null = llvm.mlir.zero : !llvm.ptr
      %one_i32 = hw.constant 1 : i32
      %delay = llvm.mlir.constant(5 : i64) : i64
      %flagAddr = llvm.mlir.addressof @flag : !llvm.ptr
      %one = llvm.mlir.constant(1 : i32) : i32

      llvm.call @__moore_delay(%delay) : (i64) -> ()
      llvm.store %one, %flagAddr : i32, !llvm.ptr
      func.call @"uvm_pkg::uvm_phase::drop_objection"(%phase, %null, %null, %one_i32)
        : (i64, !llvm.ptr, !llvm.ptr, i32) -> ()
      llhd.halt
    }

    hw.output
  }
}

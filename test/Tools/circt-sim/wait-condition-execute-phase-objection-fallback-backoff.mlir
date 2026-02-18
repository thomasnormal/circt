// RUN: env CIRCT_SIM_TRACE_WAIT_CONDITION=1 circt-sim %s --max-time=2000000 --max-deltas=8 2>&1 | FileCheck %s
//
// Verify execute_phase wait(condition) uses sparse objection fallback polling.
// The primary wakeup path is objection-zero waiters; timed polling is only
// a watchdog and should be scheduled at 1 us intervals.
//
// CHECK: [WAITCOND]
// CHECK-SAME: func=uvm_pkg::uvm_phase_hopper::execute_phase
// CHECK-SAME: objectionWaitHandle=
// CHECK-SAME: targetTimeFs=1000000000

module {
  llvm.func @__moore_wait_condition(i32)
  func.func @"uvm_pkg::uvm_phase::get_objection"(%phase: i64) -> !llvm.ptr {
    %null = llvm.mlir.zero : !llvm.ptr
    return %null : !llvm.ptr
  }

  func.func @"uvm_pkg::uvm_phase_hopper::execute_phase"(%self: i64, %phase: i64) {
    %c0 = hw.constant 0 : i32
    llvm.call @__moore_wait_condition(%c0) : (i32) -> ()
    return
  }

  hw.module @top() {
    llhd.process {
      %self = llvm.mlir.constant(1 : i64) : i64
      %dummy = llvm.mlir.constant(2048 : i64) : i64
      %phase = llvm.mlir.constant(4096 : i64) : i64
      %unused = func.call @"uvm_pkg::uvm_phase::get_objection"(%dummy) : (i64) -> !llvm.ptr
      func.call @"uvm_pkg::uvm_phase_hopper::execute_phase"(%self, %phase) : (i64, i64) -> ()
      llhd.halt
    }
    hw.output
  }
}

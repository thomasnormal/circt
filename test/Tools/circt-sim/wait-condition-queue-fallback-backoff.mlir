// RUN: env CIRCT_SIM_TRACE_WAIT_CONDITION=1 circt-sim %s --max-time=2000000 --max-deltas=8 2>&1 | FileCheck %s
//
// Verify queue-backed wait(condition) uses sparse fallback polling.
// Queue mutation wakeups are the primary mechanism; timed polling is only
// a watchdog and should schedule at 1 us intervals.
//
// CHECK: [WAITCOND]
// CHECK-SAME: queueWait=0x
// CHECK-SAME: targetTimeFs=1000000000

module {
  llvm.func @__moore_wait_condition(i32)
  llvm.func @__moore_queue_size(i64) -> i32

  func.func @queue_wait(%queue: i64) {
    %size = llvm.call @__moore_queue_size(%queue) : (i64) -> i32
    %c0 = hw.constant 0 : i32
    %cond = comb.icmp ne %size, %c0 : i32
    %condI32 = llvm.zext %cond : i1 to i32
    llvm.call @__moore_wait_condition(%condI32) : (i32) -> ()
    return
  }

  hw.module @top() {
    llhd.process {
      %queue = llvm.mlir.constant(4096 : i64) : i64
      func.call @queue_wait(%queue) : (i64) -> ()
      llhd.halt
    }
    hw.output
  }
}

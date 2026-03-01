// RUN: circt-sim %s --max-deltas=64 --max-time=1000000 2>&1 | FileCheck %s
// RUN: circt-sim %s --max-deltas=64 --max-time=1000000 --max-process-steps=40 2>&1 | FileCheck %s --check-prefix=CHECK-STEPS
//
// Regression: if wait_for_waiters polling at time 0 is scheduled on nextDelta
// instead of advancing real time, the simulation can churn deltas forever at
// 0 fs and never reach delayed events.
//
// CHECK-NOT: ERROR(DELTA_OVERFLOW)
// CHECK: [circt-sim] Simulation completed at time 100000 fs
// CHECK-STEPS-NOT: ERROR(PROCESS_STEP_OVERFLOW)
// CHECK-STEPS: [circt-sim] Simulation completed at time 100000 fs

module {
  llvm.func @__moore_delay(i64)

  func.func @"uvm_pkg::uvm_phase_hopper::wait_for_waiters"(
      %self: !llvm.ptr, %phase: !llvm.ptr, %state: i32) {
    %zero = llvm.mlir.constant(0 : i64) : i64
    llvm.call @__moore_delay(%zero) : (i64) -> ()
    return
  }

  hw.module @top() {
    %null = llvm.mlir.zero : !llvm.ptr
    %zero32 = llvm.mlir.constant(0 : i32) : i32
    %t100ps = llhd.constant_time <100ps, 0d, 0e>

    // Call wait_for_waiters through direct func.call in a tight loop from time
    // 0. This hits the runtime wait_for_waiters interception path.
    llhd.process {
      cf.br ^loop
    ^loop:
      func.call @"uvm_pkg::uvm_phase_hopper::wait_for_waiters"(%null, %null, %zero32)
          : (!llvm.ptr, !llvm.ptr, i32) -> ()
      cf.br ^loop
    }

    // Bound the test at 100 ps. This requires real-time progress from t=0.
    llhd.process {
      llhd.wait delay %t100ps, ^bb1
    ^bb1:
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}

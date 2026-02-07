// RUN: circt-sim %s --top=test_wait_condition_func --sim-stats --max-time=10000000 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test __moore_wait_condition called from inside a function.
// This pattern occurs in UVM when uvm_phase_hopper::get() polls a queue
// via wait_condition(queue.size != 0). The wait_condition is inside the
// function body, called from an llhd.process via llvm.call.
//
// The key behaviors being tested:
// 1. The restart block must be the FUNCTION body's block (not the process block)
// 2. Call stack frame resume position must be overridden to re-evaluate condition
// 3. The arith ops in the condition chain (trunci, icmp) must be traced through
//
// wait_condition uses 1 ps polling, so the condition is re-checked at 1 ps.

// CHECK: [circt-sim] Found 2 LLHD processes
// CHECK: [circt-sim] Starting simulation
// Proc 1 delays 5 fs then writes 1 to the counter global.
// Proc 2 calls wait_for_nonzero() which polls the counter via wait_condition.
// wait_condition polls every 1 ps (1000000 fs), so condition is re-checked at 1 ps.
// CHECK: [circt-sim] Simulation completed at time 1000000 fs
// CHECK: [circt-sim] Simulation completed

// External declarations
llvm.func @__moore_wait_condition(i32)

// Global counter (simulates a queue size field in a struct).
// Starts at 0; another process will write 1 to it after a delay.
llvm.mlir.global internal @counter(0 : i64) : i64

// Function that polls the counter via wait_condition.
// This mirrors the pattern in uvm_phase_hopper::get():
//   load counter → trunci → icmp ne 0 → zext → wait_condition
llvm.func @wait_for_nonzero() {
  %addr = llvm.mlir.addressof @counter : !llvm.ptr
  %val = llvm.load %addr : !llvm.ptr -> i64
  %trunc = arith.trunci %val : i64 to i32
  %c0 = arith.constant 0 : i32
  %ne = comb.icmp bin ne %trunc, %c0 : i32
  %cond = llvm.zext %ne : i1 to i32
  llvm.call @__moore_wait_condition(%cond) : (i32) -> ()
  llvm.return
}

hw.module @test_wait_condition_func() {
  %c0_i8 = hw.constant 0 : i8
  %c1_i8 = hw.constant 1 : i8
  %delta = llhd.constant_time <0ns, 1d, 0e>
  %delay5 = llhd.constant_time <5fs, 0d, 0e>

  %sig = llhd.sig %c0_i8 : i8

  // Process 1: After 5 fs, write 1 to the global counter.
  // This simulates try_put() pushing an element into the phase queue.
  llhd.process {
    llhd.wait delay %delay5, ^write
  ^write:
    %addr = llvm.mlir.addressof @counter : !llvm.ptr
    %one = arith.constant 1 : i64
    llvm.store %one, %addr : i64, !llvm.ptr
    llhd.halt
  }

  // Process 2: Call wait_for_nonzero() which blocks until counter != 0.
  // This simulates the phase hopper daemon calling get().
  // After the condition becomes true, drive the output signal.
  llhd.process {
    llvm.call @wait_for_nonzero() : () -> ()
    // Condition was met — drive signal to indicate success
    llhd.drv %sig, %c1_i8 after %delta : i8
    llhd.halt
  }

  hw.output
}

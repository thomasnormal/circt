// RUN: circt-sim %s --top=test_llvm_memory --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test LLVM dialect memory operations (alloca, load, store) in LLHD processes.
// This is essential for class simulation support.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 0 fs
// CHECK: Processes executed:   1
// CHECK: Signal updates:       2
// CHECK: [circt-sim] Simulation completed

hw.module @test_llvm_memory() {
  %c0_i32 = hw.constant 0 : i32
  %c42_i32 = hw.constant 42 : i32
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i32 : i32

  llhd.process {
    // Allocate memory for a 32-bit integer
    %c1_i64 = arith.constant 1 : i64
    %ptr = llvm.alloca %c1_i64 x i32 : (i64) -> !llvm.ptr

    // Store a value
    llvm.store %c42_i32, %ptr : i32, !llvm.ptr

    // Load it back
    %loaded = llvm.load %ptr : !llvm.ptr -> i32

    // Drive the signal with the loaded value
    llhd.drv %sig, %loaded after %delta : i32
    llhd.halt
  }

  hw.output
}

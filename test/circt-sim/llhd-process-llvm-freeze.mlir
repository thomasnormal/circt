// RUN: circt-sim %s --top=test_llvm_freeze --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test LLVM freeze operation in LLHD processes.
// llvm.freeze converts undefined values (poison/undef) to deterministic values.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 0 fs
// CHECK: Processes executed:   1
// CHECK: Signal updates:       2
// CHECK: [circt-sim] Simulation completed

hw.module @test_llvm_freeze() {
  %c0_i32 = hw.constant 0 : i32
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i32 : i32

  llhd.process {
    // Test freeze on a known value (should pass through)
    %c42 = arith.constant 42 : i32
    %frozen1 = llvm.freeze %c42 : i32  // Should be 42

    // Test freeze on undef (should become 0 in our implementation)
    %undef = llvm.mlir.undef : i32
    %frozen2 = llvm.freeze %undef : i32  // Should be 0 (deterministic)

    // Result: 42 + 0 = 42
    %result = llvm.add %frozen1, %frozen2 : i32

    llhd.drv %sig, %result after %delta : i32
    llhd.halt
  }

  hw.output
}

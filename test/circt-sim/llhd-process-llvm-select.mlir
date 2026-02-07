// RUN: circt-sim %s --top=test_llvm_select --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test LLVM select operation in LLHD processes.
// llvm.select conditionally selects between two values based on a boolean condition.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 0 fs
// CHECK: Processes executed:   1
// CHECK: Signal updates:       2
// CHECK: [circt-sim] Simulation completed

hw.module @test_llvm_select() {
  %c0_i32 = hw.constant 0 : i32
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i32 : i32

  llhd.process {
    %c10 = arith.constant 10 : i32
    %c20 = arith.constant 20 : i32

    // Test select with true condition
    %true = arith.constant 1 : i1
    %sel1 = llvm.select %true, %c10, %c20 : i1, i32  // Should be 10

    // Test select with false condition
    %false = arith.constant 0 : i1
    %sel2 = llvm.select %false, %c10, %c20 : i1, i32  // Should be 20

    // Combine: 10 + 20 = 30
    %result = llvm.add %sel1, %sel2 : i32

    llhd.drv %sig, %result after %delta : i32
    llhd.halt
  }

  hw.output
}

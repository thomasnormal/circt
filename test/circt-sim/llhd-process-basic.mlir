// RUN: circt-sim %s --top=test_basic --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test basic LLHD process execution without delays.
// The process should:
// 1. Start at time 0
// 2. Drive a signal with a delta delay
// 3. Halt

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 0 fs
// CHECK: Processes executed: 1
// CHECK: Signal updates: 2
// CHECK: [circt-sim] Simulation completed

hw.module @test_basic() {
  %c0_i8 = hw.constant 0 : i8
  %c1_i8 = hw.constant 42 : i8
  %delta = llhd.constant_time <0ns, 1d, 0e>

  // Create a signal with initial value 0
  %sig = llhd.sig %c0_i8 : i8

  // Process that drives a new value with delta delay
  llhd.process {
    // Drive value 42 after delta delay
    llhd.drv %sig, %c1_i8 after %delta : i8
    llhd.halt
  }

  hw.output
}

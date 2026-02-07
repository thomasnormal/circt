// RUN: circt-sim %s --top=test_probe --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test LLHD process with probe and drive operations.
// The process should:
// 1. Probe the signal initial value
// 2. Drive a new value

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 0 fs
// CHECK: Processes executed: 1
// CHECK: [circt-sim] Simulation completed

hw.module @test_probe() {
  %c0_i8 = hw.constant 0 : i8
  %c1_i8 = hw.constant 1 : i8
  %delta = llhd.constant_time <0ns, 1d, 0e>

  // Create a signal with initial value 0
  %sig = llhd.sig %c0_i8 : i8

  // Process that probes and then drives
  llhd.process {
    // Probe the current value
    %val = llhd.prb %sig : i8
    // Drive the inverse (we're just using 1 for simplicity)
    llhd.drv %sig, %c1_i8 after %delta : i8
    llhd.halt
  }

  hw.output
}

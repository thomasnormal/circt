// RUN: circt-sim %s --top=test_event_wait_no_drive --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test LLHD process with event-based wait but no drive operation.
// This process should NOT be removed by canonicalization because:
// 1. It has a wait with observed signals (sensitivity list)
// 2. It's part of the simulation semantics

// CHECK: [circt-sim] Found 2 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 2 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 1 fs
// CHECK: Processes executed:   4

hw.module @test_event_wait_no_drive() {
  %c0_i8 = hw.constant 0 : i8
  %c42_i8 = hw.constant 42 : i8
  %delta = llhd.constant_time <0ns, 1d, 0e>
  %delay = llhd.constant_time <1fs, 0d, 0e>

  // Create signal
  %sig = llhd.sig %c0_i8 : i8

  // Process 1: Drive signal after a delay
  llhd.process {
    llhd.wait delay %delay, ^drive
  ^drive:
    llhd.drv %sig, %c42_i8 after %delta : i8
    llhd.halt
  }

  // Process 2: Wait for signal to change (no drive)
  // This process should still exist and wait for events
  llhd.process {
    %val = llhd.prb %sig : i8
    llhd.wait (%val : i8), ^respond
  ^respond:
    // Intentionally no drive - just halt after receiving the event
    llhd.halt
  }

  hw.output
}

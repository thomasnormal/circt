// RUN: circt-sim %s --top=test_event_wait_internal --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test event-based wait with probe inside the process.
// This test verifies that processes with event-based waits are not
// incorrectly removed by canonicalization even when they don't have
// other visible side effects (like DriveOp).

// CHECK: [circt-sim] Found 2 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 2 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 1 fs
// CHECK: Processes executed:   4

hw.module @test_event_wait_internal() {
  %c0_i8 = hw.constant 0 : i8
  %c1_i8 = hw.constant 1 : i8
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

  // Process 2: Wait for signal to change - probe inside the process
  llhd.process {
    // Probe inside the process block
    %val = llhd.prb %sig : i8
    // Event-based wait
    llhd.wait (%val : i8), ^respond
  ^respond:
    // After signal changes, halt
    llhd.halt
  }

  hw.output
}

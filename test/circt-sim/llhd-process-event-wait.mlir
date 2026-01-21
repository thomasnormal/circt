// RUN: circt-sim %s --top=test_event_wait --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test LLHD process with event-based wait (sensitivity list).
// The process should wait for signal changes and then continue.

// CHECK: [circt-sim] Found 2 LLHD processes
// CHECK: [circt-sim] Registered 2 LLHD signals and 2 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 1 fs
// CHECK: Processes executed:   4

hw.module @test_event_wait() {
  %c0_i8 = hw.constant 0 : i8
  %c1_i8 = hw.constant 1 : i8
  %delta = llhd.constant_time <0ns, 1d, 0e>
  %delay = llhd.constant_time <1fs, 0d, 0e>

  // Create two signals
  %sigA = llhd.sig %c0_i8 : i8
  %sigB = llhd.sig %c0_i8 : i8

  // Process 1: Drive sigA after a delay to trigger the event-based wait
  llhd.process {
    llhd.wait delay %delay, ^bb1
  ^bb1:
    // Drive sigA to 1 after the delay
    llhd.drv %sigA, %c1_i8 after %delta : i8
    llhd.halt
  }

  // Process 2: Wait for sigA to change using event-based wait
  llhd.process {
    // Probe the signal
    %valA = llhd.prb %sigA : i8
    // Wait for the signal to change
    llhd.wait (%valA : i8), ^bb1
  ^bb1:
    // When sigA changes, read and copy to sigB
    %newValA = llhd.prb %sigA : i8
    llhd.drv %sigB, %newValA after %delta : i8
    llhd.halt
  }

  hw.output
}

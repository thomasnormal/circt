// RUN: circt-sim %s --top=test_process --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim
// XFAIL: *

// This test documents the current limitation: circt-sim does NOT interpret
// llhd.process bodies. The simulation infrastructure (ProcessScheduler, 
// EventQueue) exists, but the connection to LLHD IR is not implemented.
//
// Current behavior: Simulation ends at 0fs with placeholder process only.
// Expected behavior: Should execute llhd.process body with llhd.wait delays.
//
// See PROJECT_PLAN.md Track A for detailed analysis.

// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 10000000 fs
// CHECK: Processes executed: 1

hw.module @test_process() {
  %c0_i8 = hw.constant 0 : i8
  %c1_i8 = hw.constant 1 : i8
  %delay = llhd.constant_time <10000000fs, 0d, 0e>
  %epsilon = llhd.constant_time <0ns, 0d, 1e>
  
  // Create a signal
  %sig = llhd.sig %c0_i8 : i8
  
  // Process that should drive the signal after a delay
  llhd.process {
    // Wait 10ms (10000000fs)
    llhd.wait delay %delay, ^bb1
  ^bb1:
    // Drive value after epsilon delay
    llhd.drv %sig, %c1_i8 after %epsilon : i8
    llhd.halt
  }
  
  hw.output
}

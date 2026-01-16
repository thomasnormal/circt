// RUN: circt-sim %s --top=test_process --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// This test verifies that circt-sim correctly interprets LLHD process bodies
// with llhd.wait delays and llhd.drv operations.
//
// The process:
// 1. Starts at time 0
// 2. Waits for 10000000 fs (10 ms)
// 3. Drives the signal with a new value
// 4. Halts

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 10000000 fs
// CHECK: Processes executed: 2

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

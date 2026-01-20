// RUN: circt-sim %s --top=test_wait_probe --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test LLHD process with wait + probe. Should advance time, but currently
// finishes at time 1 fs.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 1 fs

hw.module @test_wait_probe() {
  %c0_i8 = hw.constant 0 : i8
  %c1_i8 = hw.constant 1 : i8
  %delay = llhd.constant_time <1fs, 0d, 0e>
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i8 : i8

  llhd.process {
    llhd.wait delay %delay, ^bb1
  ^bb1:
    %val = llhd.prb %sig : i8
    llhd.drv %sig, %val after %delta : i8
    llhd.halt
  }

  hw.output
}

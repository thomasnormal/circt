// RUN: circt-sim %s --top=test_reverse --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Exercise comb reverse support in LLHD processes.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 1 fs
// CHECK: Processes executed: 2

hw.module @test_reverse() {
  %c0_i4 = hw.constant 0 : i4
  %c9_i4 = hw.constant 9 : i4
  %delay = llhd.constant_time <1fs, 0d, 0e>
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i4 : i4

  llhd.process {
    llhd.wait delay %delay, ^bb1
  ^bb1:
    %rev = comb.reverse %c9_i4 : i4
    llhd.drv %sig, %rev after %delta : i4
    llhd.halt
  }

  hw.output
}

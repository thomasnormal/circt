// RUN: circt-sim %s --top=test_parity --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Exercise comb parity support in LLHD processes.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 1 fs
// CHECK: Processes executed: 2

hw.module @test_parity() {
  %c0_i1 = hw.constant 0 : i1
  %c5_i4 = hw.constant 5 : i4
  %delay = llhd.constant_time <1fs, 0d, 0e>
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i1 : i1

  llhd.process {
    llhd.wait delay %delay, ^bb1
  ^bb1:
    %par = comb.parity %c5_i4 : i4
    llhd.drv %sig, %par after %delta : i1
    llhd.halt
  }

  hw.output
}

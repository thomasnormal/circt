// RUN: circt-sim %s --top=test_branch --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test a simple LLHD process with a conditional branch.
// NOTE: circt-sim currently completes at time 0 fs.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 0 fs

hw.module @test_branch() {
  %c0_i1 = hw.constant 0 : i1
  %c1_i1 = hw.constant 1 : i1
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i1 : i1

  llhd.process {
    %cond = comb.icmp bin eq %c1_i1, %c1_i1 : i1
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    llhd.drv %sig, %c1_i1 after %delta : i1
    llhd.halt
  ^bb2:
    llhd.drv %sig, %c0_i1 after %delta : i1
    llhd.halt
  }

  hw.output
}

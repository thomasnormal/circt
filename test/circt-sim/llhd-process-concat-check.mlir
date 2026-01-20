// RUN: circt-sim %s --top=test_concat_check --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Validate concat operand ordering by branching on an expected value.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 2 fs
// CHECK: Processes executed: 3

hw.module @test_concat_check() {
  %c0_i1 = hw.constant 0 : i1
  %c1_i1 = hw.constant 1 : i1
  %c2_i2 = hw.constant 2 : i2
  %c0_i2 = hw.constant 0 : i2
  %delay = llhd.constant_time <1fs, 0d, 0e>
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i2 : i2

  llhd.process {
    llhd.wait delay %delay, ^bb1
  ^bb1:
    %concat = comb.concat %c1_i1, %c0_i1 : i1, i1
    %cmp = comb.icmp bin eq %concat, %c2_i2 : i2
    cf.cond_br %cmp, ^bb2, ^bb3
  ^bb2:
    llhd.wait delay %delay, ^bb4
  ^bb4:
    llhd.drv %sig, %concat after %delta : i2
    llhd.halt
  ^bb3:
    llhd.drv %sig, %c0_i2 after %delta : i2
    llhd.halt
  }

  hw.output
}

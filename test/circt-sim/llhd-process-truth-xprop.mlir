// RUN: circt-sim %s --top=test_truth_xprop --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Exercise truth_table X-prop when all matching entries are identical.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 2 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 2 fs
// CHECK: Processes executed: 3

hw.module @test_truth_xprop() {
  %c0_i1 = hw.constant 0 : i1
  %c1_i1 = hw.constant 1 : i1
  %delay = llhd.constant_time <1fs, 0d, 0e>
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %init = comb.add %c0_i1, %c0_i1 : i1
  %xsig = llhd.sig %init : i1
  %osig = llhd.sig %c0_i1 : i1

  llhd.process {
    llhd.wait delay %delay, ^bb1
  ^bb1:
    %x = llhd.prb %xsig : i1
    %tt = comb.truth_table %x -> [true, true]
    cf.cond_br %tt, ^bb2, ^bb3
  ^bb2:
    llhd.wait delay %delay, ^bb4
  ^bb4:
    llhd.drv %osig, %c1_i1 after %delta : i1
    llhd.halt
  ^bb3:
    llhd.halt
  }

  hw.output
}

// RUN: circt-sim %s --top=test_truth_repl --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Exercise comb truth_table and replicate support in LLHD processes.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 1 fs
// CHECK: Processes executed: 2

hw.module @test_truth_repl() {
  %c0_i1 = hw.constant 0 : i1
  %c1_i1 = hw.constant 1 : i1
  %c0_i4 = hw.constant 0 : i4
  %delay = llhd.constant_time <1fs, 0d, 0e>
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i4 : i4

  llhd.process {
    llhd.wait delay %delay, ^bb1
  ^bb1:
    %tt = comb.truth_table %c1_i1, %c0_i1 -> [false, true, true, false]
    %rep = comb.replicate %c1_i1 : (i1) -> i4
    %val = comb.mux %tt, %rep, %c0_i4 : i4
    llhd.drv %sig, %val after %delta : i4
    llhd.halt
  }

  hw.output
}

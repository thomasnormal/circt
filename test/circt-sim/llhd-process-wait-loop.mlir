// RUN: circt-sim %s --top=test_wait_loop --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test LLHD process with a simple wait loop.
// Should advance time twice to 2 fs.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 2 fs
// CHECK: Processes executed: 3

hw.module @test_wait_loop() {
  %c0_i8 = hw.constant 0 : i8
  %c1_i8 = hw.constant 1 : i8
  %c2_i8 = hw.constant 2 : i8
  %delay = llhd.constant_time <1fs, 0d, 0e>
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i8 : i8

  llhd.process {
    cf.br ^bb1(%c0_i8 : i8)
  ^bb1(%i: i8):
    %cmp = comb.icmp bin ult %i, %c2_i8 : i8
    cf.cond_br %cmp, ^bb2, ^bb3
  ^bb2:
    llhd.wait delay %delay, ^bb4
  ^bb4:
    llhd.drv %sig, %i after %delta : i8
    %next = comb.add %i, %c1_i8 : i8
    cf.br ^bb1(%next : i8)
  ^bb3:
    llhd.halt
  }

  hw.output
}

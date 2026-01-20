// RUN: circt-sim %s --top=test_concat_multi --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Exercise multi-operand comb concat in LLHD processes.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 1 fs
// CHECK: Processes executed: 2

hw.module @test_concat_multi() {
  %c0_i6 = hw.constant 0 : i6
  %c1_i2 = hw.constant 1 : i2
  %c5_i3 = hw.constant 5 : i3
  %c1_i1 = hw.constant 1 : i1
  %delay = llhd.constant_time <1fs, 0d, 0e>
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i6 : i6

  llhd.process {
    llhd.wait delay %delay, ^bb1
  ^bb1:
    %concat = comb.concat %c1_i2, %c5_i3, %c1_i1 : i2, i3, i1
    %rev = comb.reverse %concat : i6
    llhd.drv %sig, %rev after %delta : i6
    llhd.halt
  }

  hw.output
}

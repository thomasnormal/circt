// RUN: circt-sim %s --top=test_extract_concat --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Exercise comb extract and concat support in LLHD processes.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 1 fs
// CHECK: Processes executed: 2

hw.module @test_extract_concat() {
  %c0_i8 = hw.constant 0 : i8
  %c9_i8 = hw.constant 9 : i8
  %c2_i8 = hw.constant 2 : i8
  %delay = llhd.constant_time <1fs, 0d, 0e>
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i8 : i8

  llhd.process {
    llhd.wait delay %delay, ^bb1
  ^bb1:
    %hi = comb.extract %c9_i8 from 2 : (i8) -> i3
    %lo = comb.extract %c2_i8 from 0 : (i8) -> i2
    %concat = comb.concat %hi, %lo : i3, i2
    %rev = comb.reverse %concat : i5
    %ext = comb.concat %rev, %c0_i8 : i5, i8
    %slice = comb.extract %ext from 5 : (i13) -> i8
    llhd.drv %sig, %slice after %delta : i8
    llhd.halt
  }

  hw.output
}

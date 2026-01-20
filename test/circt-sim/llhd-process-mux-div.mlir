// RUN: circt-sim %s --top=test_mux_div --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Exercise comb mux and div/mod support in LLHD processes.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 1 fs
// CHECK: Processes executed: 2

hw.module @test_mux_div() {
  %c0_i8 = hw.constant 0 : i8
  %c1_i1 = hw.constant 1 : i1
  %c1_i8 = hw.constant 1 : i8
  %c2_i8 = hw.constant 2 : i8
  %c3_i8 = hw.constant 3 : i8
  %c7_i8 = hw.constant 7 : i8
  %c8_i8 = hw.constant 8 : i8
  %delay = llhd.constant_time <1fs, 0d, 0e>
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i8 : i8

  llhd.process {
    llhd.wait delay %delay, ^bb1
  ^bb1:
    %divu = comb.divu %c8_i8, %c2_i8 : i8
    %modu = comb.modu %c7_i8, %c3_i8 : i8
    %divs = comb.divs %c8_i8, %c2_i8 : i8
    %mods = comb.mods %c7_i8, %c3_i8 : i8
    %mux = comb.mux %c1_i1, %divu, %modu : i8
    %sum0 = comb.add %divs, %mods : i8
    %sum1 = comb.add %mux, %sum0 : i8
    llhd.drv %sig, %sum1 after %delta : i8
    llhd.halt
  }

  hw.output
}

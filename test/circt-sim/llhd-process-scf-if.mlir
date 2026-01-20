// RUN: circt-sim %s --top=test_scf_if --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test SCF if/else interpretation in LLHD processes.
// The process uses scf.if to select between two values.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 1 fs
// CHECK: Processes executed: 2

hw.module @test_scf_if() {
  %c0_i8 = hw.constant 0 : i8
  %c10_i8 = hw.constant 10 : i8
  %c20_i8 = hw.constant 20 : i8
  %c5_i8 = hw.constant 5 : i8
  %delay = llhd.constant_time <1fs, 0d, 0e>
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i8 : i8

  llhd.process {
    llhd.wait delay %delay, ^bb1
  ^bb1:
    // Test scf.if with condition true: 10 > 5
    %cond = arith.cmpi ugt, %c10_i8, %c5_i8 : i8
    %result = scf.if %cond -> (i8) {
      scf.yield %c10_i8 : i8
    } else {
      scf.yield %c20_i8 : i8
    }
    llhd.drv %sig, %result after %delta : i8
    llhd.halt
  }

  hw.output
}

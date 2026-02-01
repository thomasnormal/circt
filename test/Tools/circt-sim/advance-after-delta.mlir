// RUN: circt-sim %s --sim-stats | FileCheck %s

// CHECK: [circt-sim] Simulation completed at time 1000000 fs

hw.module @AdvanceAfterDelta() {
  %false = hw.constant false
  %true = hw.constant true
  %c1_i64 = hw.constant 1000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %sig = llhd.sig %false : i1

  llhd.process {
    llhd.drv %sig, %true after %eps : i1
    llhd.halt
  }

  llhd.process {
    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}

// RUN: circt-sim --sim-stats --process-stats --process-stats-top=1 %s | FileCheck %s

// CHECK: === Process Stats
// CHECK: llhd_process_0
// CHECK: steps=

hw.module @test() {
  %false = hw.constant false
  %true = hw.constant true
  %eps = llhd.constant_time <0ns, 0d, 1e>

  // Process with more ops (should rank first).
  llhd.process {
    %v0 = comb.xor %true, %false : i1
    %v1 = comb.xor %v0, %true : i1
    %v2 = comb.xor %v1, %false : i1
    %v3 = comb.xor %v2, %true : i1
    %v4 = comb.xor %v3, %false : i1
    llhd.wait delay %eps, ^bb1
  ^bb1:
    sim.terminate success, quiet
    llhd.halt
  }

  // Process with fewer ops.
  llhd.process {
    %v0 = comb.xor %true, %false : i1
    llhd.wait delay %eps, ^bb1
  ^bb1:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}

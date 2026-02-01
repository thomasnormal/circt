// RUN: circt-sim --sim-stats --op-stats --op-stats-top=2 %s | FileCheck %s

// CHECK: === Op Stats
// CHECK: llhd.wait:
// CHECK: llhd.int_to_time:

hw.module @test() {
  %false = hw.constant false
  %true = hw.constant true
  %c1_i64 = hw.constant 1000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>

  llhd.process {
    %v0 = comb.xor %true, %false : i1
    %v1 = comb.xor %v0, %true : i1
    %v2 = comb.xor %v1, %false : i1
    llhd.wait delay %eps, ^bb1
  ^bb1:
    %v3 = comb.xor %v2, %true : i1
    %v4 = comb.xor %v3, %false : i1
    %t1 = llhd.int_to_time %c1_i64
    llhd.wait delay %t1, ^bb2
  ^bb2:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}

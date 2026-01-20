// RUN: circt-sim %s --top=test_arith --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Exercise basic comb arithmetic and bitwise operations in LLHD processes.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 1 fs
// CHECK: Processes executed: 2

hw.module @test_arith() {
  %c0_i8 = hw.constant 0 : i8
  %c1_i8 = hw.constant 1 : i8
  %c2_i8 = hw.constant 2 : i8
  %c3_i8 = hw.constant 3 : i8
  %c4_i8 = hw.constant 4 : i8
  %delay = llhd.constant_time <1fs, 0d, 0e>
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i8 : i8

  llhd.process {
    llhd.wait delay %delay, ^bb1
  ^bb1:
    %and = comb.and %c1_i8, %c2_i8 : i8
    %or = comb.or %c1_i8, %c2_i8 : i8
    %xor = comb.xor %c1_i8, %c2_i8 : i8
    %sub = comb.sub %c3_i8, %c1_i8 : i8
    %mul = comb.mul %c2_i8, %c3_i8 : i8
    %shl = comb.shl %c1_i8, %c1_i8 : i8
    %shru = comb.shru %c2_i8, %c1_i8 : i8
    %shrs = comb.shrs %c4_i8, %c1_i8 : i8
    %sum0 = comb.add %and, %or : i8
    %sum1 = comb.add %xor, %sub : i8
    %sum2 = comb.add %mul, %shl : i8
    %sum3 = comb.add %shru, %shrs : i8
    %sum4 = comb.add %sum0, %sum1 : i8
    %sum5 = comb.add %sum2, %sum3 : i8
    %sum6 = comb.add %sum4, %sum5 : i8
    llhd.drv %sig, %sum6 after %delta : i8
    llhd.halt
  }

  hw.output
}

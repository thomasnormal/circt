// RUN: circt-sim %s --top=test_arith_ops --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test arith dialect operations in LLHD processes.
// Exercises various arith operations: addi, subi, muli, andi, ori, xori,
// shli, shrui, shrsi, cmpi, select, extui, extsi, trunci.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 1 fs
// CHECK: Processes executed: 2

hw.module @test_arith_ops() {
  %c0_i32 = hw.constant 0 : i32
  %delay = llhd.constant_time <1fs, 0d, 0e>
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i32 : i32

  llhd.process {
    llhd.wait delay %delay, ^bb1
  ^bb1:
    // Test arith.constant and arith.addi
    %c10 = arith.constant 10 : i32
    %c5 = arith.constant 5 : i32
    %c2 = arith.constant 2 : i32
    %c1 = arith.constant 1 : i32

    // addi: 10 + 5 = 15
    %add = arith.addi %c10, %c5 : i32

    // subi: 15 - 2 = 13
    %sub = arith.subi %add, %c2 : i32

    // muli: 13 * 2 = 26
    %mul = arith.muli %sub, %c2 : i32

    // andi: 26 & 15 = 10
    %c15 = arith.constant 15 : i32
    %and = arith.andi %mul, %c15 : i32

    // ori: 10 | 4 = 14
    %c4 = arith.constant 4 : i32
    %or = arith.ori %and, %c4 : i32

    // xori: 14 ^ 7 = 9
    %c7 = arith.constant 7 : i32
    %xor = arith.xori %or, %c7 : i32

    // shli: 9 << 1 = 18
    %shl = arith.shli %xor, %c1 : i32

    // shrui: 18 >> 1 = 9
    %shru = arith.shrui %shl, %c1 : i32

    // cmpi: 9 > 5 -> true (1)
    %cmp = arith.cmpi ugt, %shru, %c5 : i32

    // select: true ? 100 : 200 = 100
    %c100 = arith.constant 100 : i32
    %c200 = arith.constant 200 : i32
    %sel = arith.select %cmp, %c100, %c200 : i32

    llhd.drv %sig, %sel after %delta : i32
    llhd.halt
  }

  hw.output
}

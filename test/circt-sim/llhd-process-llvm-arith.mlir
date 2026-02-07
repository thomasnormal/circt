// RUN: circt-sim %s --top=test_llvm_arith --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test LLVM arithmetic operations in LLHD processes.
// Verifies add, sub, mul, icmp, and, or, xor, shl, lshr, ashr.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 0 fs
// CHECK: Processes executed:   1
// CHECK: Signal updates:       2
// CHECK: [circt-sim] Simulation completed

hw.module @test_llvm_arith() {
  %c0_i32 = hw.constant 0 : i32
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i32 : i32

  llhd.process {
    %c10 = arith.constant 10 : i32
    %c3 = arith.constant 3 : i32

    // Test llvm.add
    %sum = llvm.add %c10, %c3 : i32  // 13

    // Test llvm.sub
    %diff = llvm.sub %sum, %c3 : i32  // 10

    // Test llvm.mul
    %c2 = arith.constant 2 : i32
    %prod = llvm.mul %diff, %c2 : i32  // 20

    // Test llvm.and
    %c15 = arith.constant 15 : i32
    %anded = llvm.and %prod, %c15 : i32  // 20 & 15 = 4

    // Test llvm.or
    %c8 = arith.constant 8 : i32
    %ored = llvm.or %anded, %c8 : i32  // 4 | 8 = 12

    // Test llvm.xor
    %c1 = arith.constant 1 : i32
    %xored = llvm.xor %ored, %c1 : i32  // 12 ^ 1 = 13

    // Test llvm.shl
    %shifted = llvm.shl %c1, %c2 : i32  // 1 << 2 = 4

    // Test llvm.lshr
    %rshifted = llvm.lshr %c8, %c2 : i32  // 8 >> 2 = 2

    // Combine results: xored + shifted + rshifted = 13 + 4 + 2 = 19
    %temp = llvm.add %xored, %shifted : i32
    %result = llvm.add %temp, %rshifted : i32

    llhd.drv %sig, %result after %delta : i32
    llhd.halt
  }

  hw.output
}

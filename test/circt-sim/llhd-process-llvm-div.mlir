// RUN: circt-sim %s --top=test_llvm_div --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test LLVM integer division and remainder operations in LLHD processes.
// Verifies sdiv, udiv, srem, urem.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 0 fs
// CHECK: Processes executed:   1
// CHECK: Signal updates:       2
// CHECK: [circt-sim] Simulation completed

hw.module @test_llvm_div() {
  %c0_i32 = hw.constant 0 : i32
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i32 : i32

  llhd.process {
    %c20 = arith.constant 20 : i32
    %c7 = arith.constant 7 : i32

    // Test unsigned division: 20 / 7 = 2
    %udiv_result = llvm.udiv %c20, %c7 : i32

    // Test signed division: 20 / 7 = 2
    %sdiv_result = llvm.sdiv %c20, %c7 : i32

    // Test unsigned remainder: 20 % 7 = 6
    %urem_result = llvm.urem %c20, %c7 : i32

    // Test signed remainder: 20 % 7 = 6
    %srem_result = llvm.srem %c20, %c7 : i32

    // Combine: 2 + 2 + 6 + 6 = 16
    %temp1 = llvm.add %udiv_result, %sdiv_result : i32
    %temp2 = llvm.add %urem_result, %srem_result : i32
    %result = llvm.add %temp1, %temp2 : i32

    llhd.drv %sig, %result after %delta : i32
    llhd.halt
  }

  hw.output
}

// RUN: circt-sim %s --top=test_moore_delay_multi --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test multiple __moore_delay calls to verify cumulative time advancement.
// This tests the pattern used in UVM phases with sequential delays.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// Expect total time = 10 + 20 + 30 = 60 fs
// CHECK: [circt-sim] Simulation completed at time 60 fs
// CHECK: [circt-sim] Simulation completed

// External declaration of __moore_delay
llvm.func @__moore_delay(i64)

// Class method with multiple delays
llvm.func @run_phases() {
  // Phase 1: delay 10 fs
  %delay1 = arith.constant 10 : i64
  llvm.call @__moore_delay(%delay1) : (i64) -> ()

  // Phase 2: delay 20 fs
  %delay2 = arith.constant 20 : i64
  llvm.call @__moore_delay(%delay2) : (i64) -> ()

  // Phase 3: delay 30 fs
  %delay3 = arith.constant 30 : i64
  llvm.call @__moore_delay(%delay3) : (i64) -> ()

  llvm.return
}

hw.module @test_moore_delay_multi() {
  %c0_i32 = hw.constant 0 : i32
  %c1_i32 = hw.constant 1 : i32
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i32 : i32

  llhd.process {
    // Call the class method with multiple delays
    llvm.call @run_phases() : () -> ()

    // After all delays, drive the signal
    llhd.drv %sig, %c1_i32 after %delta : i32
    llhd.halt
  }

  hw.output
}

// RUN: circt-sim %s --top=test_hw_array --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test hw array operations in LLHD processes.
// Exercises hw.array_create, hw.array_get, hw.array_slice.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 1 fs
// CHECK: Processes executed: 2

hw.module @test_hw_array() {
  %c0_i8 = hw.constant 0 : i8
  %delay = llhd.constant_time <1fs, 0d, 0e>
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i8 : i8

  llhd.process {
    llhd.wait delay %delay, ^bb1
  ^bb1:
    // Create an array of 4 i8 elements
    %c10 = hw.constant 10 : i8
    %c20 = hw.constant 20 : i8
    %c30 = hw.constant 30 : i8
    %c40 = hw.constant 40 : i8
    %arr = hw.array_create %c10, %c20, %c30, %c40 : i8

    // Get element at index 2 (should be 30)
    %idx = hw.constant 2 : i2
    %elem = hw.array_get %arr[%idx] : !hw.array<4xi8>, i2

    llhd.drv %sig, %elem after %delta : i8
    llhd.halt
  }

  hw.output
}

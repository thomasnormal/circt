// RUN: circt-sim %s --max-time=1000 2>&1 | FileCheck %s
// Test dynamic array element drive and probe via llhd.sig.array_get

// CHECK: [circt-sim] Simulation completed

hw.module @top() {
  // Create a 4-element array signal with initial values [0, 0, 0, 0]
  %init = hw.constant 0 : i8
  %arr_init = hw.array_create %init, %init, %init, %init : i8
  %arr = llhd.sig %arr_init : !hw.array<4xi8>

  // Process to test dynamic array access
  llhd.process {
    // Get element at dynamic index (i2 for 4-element array)
    %idx = arith.constant 2 : i2
    %elem_ref = llhd.sig.array_get %arr[%idx] : <!hw.array<4xi8>>

    // Probe initial value - should be 0
    %val0 = llhd.prb %elem_ref : i8

    // Drive a new value with epsilon delay
    %new_val = arith.constant 42 : i8
    %eps = llhd.constant_time <0ns, 1d, 0e>
    llhd.drv %elem_ref, %new_val after %eps : i8

    // Wait for the drive to take effect
    llhd.wait delay %eps, ^after_drive

  ^after_drive:
    // Probe the updated value - should be 42
    %val1 = llhd.prb %elem_ref : i8

    // Verify the value is correct (42 = 0x2a)
    %expected = arith.constant 42 : i8
    %match = arith.cmpi eq, %val1, %expected : i8

    // Finish simulation
    sim.terminate success, quiet
    llhd.halt
  }
}

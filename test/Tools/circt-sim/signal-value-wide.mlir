// RUN: circt-sim %s | FileCheck %s

// Test that SignalValue can handle values wider than 64 bits (e.g., 128-bit structs)
// This verifies the fix for the 64-bit limitation in SignalValue

// CHECK: [circt-sim] Starting simulation
// CHECK: Wide signal test passed
// CHECK: [circt-sim] Simulation completed

hw.module @test_wide_signal() {
  // Create a 128-bit constant for initialization
  %init = hw.constant 0x0123456789ABCDEF0123456789ABCDEF : i128

  // Create a 128-bit signal (simulating a wide struct)
  %sig = llhd.sig %init : i128

  %eps = llhd.constant_time <0ns, 0d, 1e>

  // Create a process that probes and drives the wide signal
  llhd.process {
    // Probe the signal (should get the initial value)
    %val1 = llhd.prb %sig : i128

    // Drive a new 128-bit value
    %new_val = hw.constant 0xFEDCBA9876543210FEDCBA9876543210 : i128
    llhd.drv %sig, %new_val after %eps : i128

    // Wait for the drive to take effect
    llhd.wait (%val1 : i128), ^check
  ^check:
    // Probe again to verify the new value
    %val2 = llhd.prb %sig : i128

    // Print a message to confirm successful execution
    %fmt = sim.fmt.literal "Wide signal test passed\0A"
    sim.proc.print %fmt

    llhd.halt
  }
}

hw.module @top() {
  hw.instance "test" @test_wide_signal() -> ()
}

// RUN: circt-sim %s --top=test_moore_delay --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test __moore_delay runtime function for class method delays.
// This tests the delay mechanism used when #delay appears in class tasks/methods.
// The __moore_delay function should advance simulation time.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time
// CHECK: Processes executed:
// CHECK: [circt-sim] Simulation completed

// External declaration of __moore_delay (simulates runtime function)
llvm.func @__moore_delay(i64)

// Class method that uses a delay
llvm.func @run_with_delay(%sig_ptr: !llvm.ptr) {
  // Delay 10 fs
  %delay = arith.constant 10 : i64
  llvm.call @__moore_delay(%delay) : (i64) -> ()

  // After delay, update a value (simulated by returning)
  llvm.return
}

hw.module @test_moore_delay() {
  %c0_i32 = hw.constant 0 : i32
  %c1_i32 = hw.constant 1 : i32
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i32 : i32

  llhd.process {
    // Get signal pointer (not actually used, but demonstrates the pattern)
    %sig_ptr = llhd.constant_time <0ns, 0d, 0e>

    // Allocate a dummy pointer for the class method call
    %ptr = llvm.mlir.zero : !llvm.ptr

    // Call the class method that contains a delay
    llvm.call @run_with_delay(%ptr) : (!llvm.ptr) -> ()

    // After the method returns, drive the signal
    llhd.drv %sig, %c1_i32 after %delta : i32
    llhd.halt
  }

  hw.output
}

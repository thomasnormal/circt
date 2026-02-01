// RUN: circt-sim %s --max-time=100000000 2>&1 | FileCheck %s
// Test wait(condition) where condition depends on heap memory (class member variable)
// This tests the polling-based re-evaluation mechanism for memory-backed conditions

// CHECK: Starting test
// CHECK: Setting up condition variable
// CHECK: Waiting for condition
// CHECK: Triggering condition change
// CHECK: Condition met!
// CHECK: Test PASSED

module {
  llvm.func @malloc(i64) -> !llvm.ptr

  hw.module @top() {
    %c0_i32 = hw.constant 0 : i32
    %c1_i32 = hw.constant 1 : i32
    %c16_i64 = hw.constant 16 : i64
    %c50000000_i64 = hw.constant 50000000 : i64

    %fmt_start = sim.fmt.literal "Starting test\0A"
    %fmt_setup = sim.fmt.literal "Setting up condition variable\0A"
    %fmt_wait = sim.fmt.literal "Waiting for condition\0A"
    %fmt_trigger = sim.fmt.literal "Triggering condition change\0A"
    %fmt_met = sim.fmt.literal "Condition met!\0A"
    %fmt_pass = sim.fmt.literal "Test PASSED\0A"

    %time_epsilon = llhd.constant_time <0ns, 0d, 1e>
    %null = llvm.mlir.zero : !llvm.ptr

    // Signal to hold condition variable pointer
    %cond_ptr_sig = llhd.sig %null : !llvm.ptr

    // Process 1: Set up condition variable and wait for it
    llhd.process {
      sim.proc.print %fmt_start
      sim.proc.print %fmt_setup

      // Allocate condition variable (simulating class member)
      %ptr = llvm.call @malloc(%c16_i64) : (i64) -> !llvm.ptr
      llvm.store %c0_i32, %ptr : i32, !llvm.ptr

      // Store pointer in signal so process 2 can access it
      llhd.drv %cond_ptr_sig, %ptr after %time_epsilon : !llvm.ptr

      // Wait for signal update
      %wait_time = llhd.constant_time <1ns, 0d, 0e>
      llhd.wait delay %wait_time, ^bb1(%ptr : !llvm.ptr)
    ^bb1(%ptr_arg: !llvm.ptr):
      sim.proc.print %fmt_wait

      // Wait for condition (value == 1) - this uses polling for memory
      %cond_val = llvm.load %ptr_arg : !llvm.ptr -> i32
      %cond = comb.icmp eq %cond_val, %c1_i32 : i32
      // Convert i1 to i32 using llvm.zext
      %cond_i32 = llvm.zext %cond : i1 to i32
      llvm.call @__moore_wait_condition(%cond_i32) : (i32) -> ()

      sim.proc.print %fmt_met
      sim.proc.print %fmt_pass
      sim.terminate success, quiet
      llhd.halt
    }

    // Process 2: Trigger condition after delay
    llhd.process {
      %delay = llhd.int_to_time %c50000000_i64
      llhd.wait delay %delay, ^bb1
    ^bb1:
      sim.proc.print %fmt_trigger
      %ptr = llhd.prb %cond_ptr_sig : !llvm.ptr
      llvm.store %c1_i32, %ptr : i32, !llvm.ptr
      llhd.halt
    }

    hw.output
  }

  llvm.func @__moore_wait_condition(i32)
}

// RUN: circt-sim %s --max-time 100000000 2>&1 | FileCheck %s

// Test that sim.fork with blocking join correctly resumes execution
// inside a function after children complete.

// This simulates the UVM run_test() pattern where:
// 1. A process calls a function (run_test)
// 2. The function creates a blocking fork
// 3. The function should continue AFTER the fork completes
// 4. Then the process halts

// Without the call stack fix, the simulation would hang or skip the code
// after the fork in the function, causing the counter to be 3 instead of 4.

// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 1000000 fs
// CHECK: [circt-sim] Simulation completed

module {
  // Global counter to track execution order
  llvm.mlir.global internal @step_counter(0 : i32) : i32

  // This function simulates run_test():
  // 1. Increment counter (step 1: before fork)
  // 2. Create blocking fork with a child that increments counter (step 2)
  // 3. Increment counter again (step 3: after fork - this is what we're testing)
  func.func @run_test_like() {
    %c1 = arith.constant 1 : i32
    %ptr = llvm.mlir.addressof @step_counter : !llvm.ptr

    // Step 1: Before fork
    %v1 = llvm.load %ptr : !llvm.ptr -> i32
    %v2 = arith.addi %v1, %c1 : i32
    llvm.store %v2, %ptr : i32, !llvm.ptr

    // Blocking fork (default join type = "join")
    %handle = sim.fork {
      // Step 2: Child increments
      %cptr = llvm.mlir.addressof @step_counter : !llvm.ptr
      %cv1 = llvm.load %cptr : !llvm.ptr -> i32
      %cv2 = arith.addi %cv1, %c1 : i32
      llvm.store %cv2, %cptr : i32, !llvm.ptr
      sim.fork.terminator
    }

    // Step 3: After fork (should execute after child completes)
    // Without the call stack fix, this code would never execute
    %v3 = llvm.load %ptr : !llvm.ptr -> i32
    %v4 = arith.addi %v3, %c1 : i32
    llvm.store %v4, %ptr : i32, !llvm.ptr

    return
  }

  hw.module @top() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    llhd.process {
      // Call the function that contains fork
      func.call @run_test_like() : () -> ()

      // Step 4: After function returns, increment again
      %c1 = arith.constant 1 : i32
      %ptr = llvm.mlir.addressof @step_counter : !llvm.ptr
      %v1 = llvm.load %ptr : !llvm.ptr -> i32
      %v2 = arith.addi %v1, %c1 : i32
      llvm.store %v2, %ptr : i32, !llvm.ptr

      // Final counter value should be 4 (steps 1,2,3,4)
      // If fork resume doesn't work, it would be 3 (missing step 3)
      llhd.wait delay %t1, ^check

    ^check:
      // Verify counter is 4
      %expected = arith.constant 4 : i32
      %final = llvm.load %ptr : !llvm.ptr -> i32
      %eq = arith.cmpi eq, %final, %expected : i32
      cf.cond_br %eq, ^success, ^fail

    ^success:
      // Test passed
      llhd.halt

    ^fail:
      // Test failed - halt
      llhd.halt
    }
    hw.output
  }
}

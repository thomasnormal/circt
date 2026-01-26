// RUN: circt-sim %s --max-time 100000000 2>&1 | FileCheck %s
// Test that deep recursion (UVM patterns) is handled gracefully with call depth protection

// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed

module {
  llvm.mlir.global internal @recursion_counter(0 : i32) : i32

  // Recursive function that would cause stack overflow without protection
  func.func @recursive_func(%depth: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i32
    %c200 = arith.constant 200 : i32

    // Increment global counter
    %ptr = llvm.mlir.addressof @recursion_counter : !llvm.ptr
    %current = llvm.load %ptr : !llvm.ptr -> i32
    %next = arith.addi %current, %c1 : i32
    llvm.store %next, %ptr : i32, !llvm.ptr

    // If depth > 200, recurse (would overflow stack without protection)
    %cond = arith.cmpi slt, %depth, %c200 : i32
    cf.cond_br %cond, ^recurse, ^done

  ^recurse:
    %next_depth = arith.addi %depth, %c1 : i32
    %result = func.call @recursive_func(%next_depth) : (i32) -> i32
    return %result : i32

  ^done:
    return %depth : i32
  }

  hw.module @top() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    llhd.process {
      cf.br ^start
    ^start:
      %c0 = arith.constant 0 : i32
      %result = func.call @recursive_func(%c0) : (i32) -> i32
      llhd.wait delay %t1, ^done
    ^done:
      llhd.halt
    }
    hw.output
  }
}

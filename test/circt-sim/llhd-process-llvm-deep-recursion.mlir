// RUN: circt-sim %s --top=test_deep_recursion --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test that circt-sim handles very deep recursive LLVM function calls.
// This test verifies that the call depth limit (100) prevents stack overflow.
// Without the limit, this would cause a stack overflow crash.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Simulation completed

// A recursive function that counts down from n to 0
// When n > 100, this will hit the call depth limit
llvm.func @deep_recursive_count(%n: i32) -> i32 {
  %zero = llvm.mlir.constant(0 : i32) : i32
  %one = llvm.mlir.constant(1 : i32) : i32
  %cmp = llvm.icmp "sle" %n, %zero : i32
  llvm.cond_br %cmp, ^exit(%zero : i32), ^recurse
^recurse:
  %n_minus_1 = llvm.sub %n, %one : i32
  %result = llvm.call @deep_recursive_count(%n_minus_1) : (i32) -> i32
  %sum = llvm.add %result, %one : i32
  llvm.br ^exit(%sum : i32)
^exit(%final: i32):
  llvm.return %final : i32
}

hw.module @test_deep_recursion() {
  %c0_i32 = hw.constant 0 : i32
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i32 : i32

  llhd.process {
    // Test with depth that exceeds the call depth limit (100)
    // Before the fix, this would cause stack overflow
    // With the fix, it hits the call depth limit and returns X
    %deep = llvm.mlir.constant(200 : i32) : i32
    %result_deep = llvm.call @deep_recursive_count(%deep) : (i32) -> i32

    // Drive the signal with the result
    llhd.drv %sig, %result_deep after %delta : i32
    llhd.halt
  }

  hw.output
}

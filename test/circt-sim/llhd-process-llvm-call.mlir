// RUN: circt-sim %s --top=test_llvm_call --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test LLVM function call operations in LLHD processes.
// This is essential for class method calls.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 0 fs
// CHECK: Processes executed:   1
// CHECK: Signal updates:       2
// CHECK: [circt-sim] Simulation completed

// LLVM function that doubles its input
llvm.func @double_value(%arg: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %result = llvm.mul %arg, %c2 : i32
  llvm.return %result : i32
}

// LLVM function that adds two numbers
llvm.func @add_values(%a: i32, %b: i32) -> i32 {
  %result = llvm.add %a, %b : i32
  llvm.return %result : i32
}

hw.module @test_llvm_call() {
  %c0_i32 = hw.constant 0 : i32
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i32 : i32

  llhd.process {
    // Call double_value(5) = 10
    %c5 = arith.constant 5 : i32
    %doubled = llvm.call @double_value(%c5) : (i32) -> i32

    // Call add_values(10, 3) = 13
    %c3 = arith.constant 3 : i32
    %sum = llvm.call @add_values(%doubled, %c3) : (i32, i32) -> i32

    // Drive the signal with the result (should be 13)
    llhd.drv %sig, %sum after %delta : i32
    llhd.halt
  }

  hw.output
}

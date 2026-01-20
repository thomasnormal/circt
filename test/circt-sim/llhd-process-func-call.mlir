// RUN: circt-sim %s --top=test_func_call --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test func.call interpretation in LLHD processes.
// The process calls a function that computes a simple arithmetic operation.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 1 fs
// CHECK: Processes executed: 2

// Function that doubles its input
func.func @double(%x: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %result = arith.muli %x, %c2 : i32
  return %result : i32
}

// Function that adds two numbers
func.func @add(%a: i32, %b: i32) -> i32 {
  %result = arith.addi %a, %b : i32
  return %result : i32
}

hw.module @test_func_call() {
  %c0_i32 = hw.constant 0 : i32
  %delay = llhd.constant_time <1fs, 0d, 0e>
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i32 : i32

  llhd.process {
    llhd.wait delay %delay, ^bb1
  ^bb1:
    // Call double(5) = 10
    %c5 = arith.constant 5 : i32
    %doubled = func.call @double(%c5) : (i32) -> i32

    // Call add(10, 3) = 13
    %c3 = arith.constant 3 : i32
    %sum = func.call @add(%doubled, %c3) : (i32, i32) -> i32

    llhd.drv %sig, %sum after %delta : i32
    llhd.halt
  }

  hw.output
}

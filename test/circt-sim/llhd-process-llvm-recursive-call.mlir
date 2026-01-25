// RUN: circt-sim %s --top=test_recursive_call --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test that circt-sim handles recursive LLVM function calls.
// This test verifies that the interpreter can handle bounded recursion.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 0 fs
// CHECK: [circt-sim] Simulation finished successfully

// A recursive function that counts down from n to 0
llvm.func @recursive_count(%n: i32) -> i32 {
  %zero = llvm.mlir.constant(0 : i32) : i32
  %one = llvm.mlir.constant(1 : i32) : i32
  %cmp = llvm.icmp "sle" %n, %zero : i32
  llvm.cond_br %cmp, ^exit(%zero : i32), ^recurse
^recurse:
  %n_minus_1 = llvm.sub %n, %one : i32
  %result = llvm.call @recursive_count(%n_minus_1) : (i32) -> i32
  %sum = llvm.add %result, %one : i32
  llvm.br ^exit(%sum : i32)
^exit(%final: i32):
  llvm.return %final : i32
}

hw.module @test_recursive_call() {
  %c0_i32 = hw.constant 0 : i32
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i32 : i32

  llhd.process {
    // Test with moderate depth (should complete normally)
    %ten = llvm.mlir.constant(10 : i32) : i32
    %result10 = llvm.call @recursive_count(%ten) : (i32) -> i32

    // Drive the signal with the result (should be 10)
    llhd.drv %sig, %result10 after %delta : i32
    llhd.halt
  }

  hw.output
}

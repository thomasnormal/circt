// RUN: circt-sim %s --top=test_scf_for --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test SCF for loop interpretation in LLHD processes.
// The process computes the sum of integers 0..3 using scf.for.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 1 fs
// CHECK: Processes executed: 2

hw.module @test_scf_for() {
  %c0_i32 = hw.constant 0 : i32
  %c0_index = arith.constant 0 : index
  %c4_index = arith.constant 4 : index
  %c1_index = arith.constant 1 : index
  %delay = llhd.constant_time <1fs, 0d, 0e>
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i32 : i32

  llhd.process {
    llhd.wait delay %delay, ^bb1
  ^bb1:
    // Compute sum = 0 + 1 + 2 + 3 = 6
    %sum = scf.for %iv = %c0_index to %c4_index step %c1_index iter_args(%acc = %c0_i32) -> (i32) {
      %iv_i32 = arith.index_cast %iv : index to i32
      %new_acc = arith.addi %acc, %iv_i32 : i32
      scf.yield %new_acc : i32
    }
    llhd.drv %sig, %sum after %delta : i32
    llhd.halt
  }

  hw.output
}

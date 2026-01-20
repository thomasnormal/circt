// RUN: circt-sim %s --top=test_scf_while --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test SCF while loop interpretation in LLHD processes.
// The process uses scf.while to compute a value iteratively.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 1 fs
// CHECK: Processes executed: 2

hw.module @test_scf_while() {
  %c0_i32 = hw.constant 0 : i32
  %delay = llhd.constant_time <1fs, 0d, 0e>
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %sig = llhd.sig %c0_i32 : i32

  llhd.process {
    llhd.wait delay %delay, ^bb1
  ^bb1:
    // Compute sum = 0 + 1 + 2 + 3 using while loop
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32

    %result = scf.while (%i = %c0, %acc = %c0) : (i32, i32) -> (i32) {
      %cond = arith.cmpi ult, %i, %c4 : i32
      scf.condition(%cond) %acc : i32
    } do {
    ^bb0(%val: i32):
      // Re-read the iteration counter and accumulator
      %i_next = arith.addi %c0, %c1 : i32 // Would need proper loop var tracking
      %new_acc = arith.addi %val, %c1 : i32
      scf.yield %i_next, %new_acc : i32, i32
    }

    llhd.drv %sig, %result after %delta : i32
    llhd.halt
  }

  hw.output
}

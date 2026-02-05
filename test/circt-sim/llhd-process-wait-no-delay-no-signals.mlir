// RUN: circt-sim %s --top=test_wait_no_delay_no_signals --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test LLHD process with wait that has NO delay AND NO observed signals.
// This represents `always @(*)` semantics - the process should resume
// on the next delta cycle instead of hanging forever.
//
// Before the fix, this would cause the process to hang indefinitely because:
// 1. No delay means no timed wakeup was scheduled
// 2. No observed signals means no signal sensitivity was registered
// 3. The process was marked as waiting but never scheduled to resume
//
// After the fix, the process schedules immediate delta-step resumption.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Registered 2 LLHD signals and 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 0 fs
// CHECK: Processes executed:   3

hw.module @test_wait_no_delay_no_signals() {
  %c0_i8 = hw.constant 0 : i8
  %c1_i8 = hw.constant 1 : i8
  %c2_i8 = hw.constant 2 : i8
  %delta = llhd.constant_time <0ns, 1d, 0e>

  // Counter signal to track iterations
  %counter = llhd.sig %c0_i8 : i8
  // Result signal to verify the test passed
  %result = llhd.sig %c0_i8 : i8

  // Process 1: Uses wait with no delay and no observed signals
  // This simulates `always @(*)` - should run twice (iterations 0 and 1)
  // and then halt, not hang forever.
  llhd.process {
    cf.br ^check
  ^check:
    %cnt = llhd.prb %counter : i8
    %done = comb.icmp bin uge %cnt, %c2_i8 : i8
    cf.cond_br %done, ^done, ^loop
  ^loop:
    // Increment counter
    %next = comb.add %cnt, %c1_i8 : i8
    llhd.drv %counter, %next after %delta : i8
    // Wait with NO delay and NO observed signals - the bug case
    // Before fix: process would hang here forever
    // After fix: process resumes on next delta cycle
    llhd.wait ^check
  ^done:
    // Mark that we successfully completed
    llhd.drv %result, %c1_i8 after %delta : i8
    llhd.halt
  }

  hw.output
}

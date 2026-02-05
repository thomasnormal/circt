// RUN: circt-sim %s --top=test_runtime_sig --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test runtime signal creation in initial blocks and processes.
// This tests the handler for llhd.sig during interpretation, which is needed
// to support local variables in global constructors and initial blocks.

// CHECK: [circt-sim] Found 0 LLHD processes, 1 seq.initial blocks
// CHECK: [circt-sim] Starting simulation
// CHECK: Initial value: 42
// CHECK: After modification: 100
// CHECK: [circt-sim] Simulation completed at time 0 fs
// CHECK: [circt-sim] Simulation finished successfully

hw.module @test_runtime_sig() {
  seq.initial() {
    %c42_i32 = hw.constant 42 : i32
    %c100_i32 = hw.constant 100 : i32
    %time = llhd.constant_time <0ns, 0d, 1e>

    // Create a runtime signal (local variable) with initial value 42
    %local_var = llhd.sig name "local_var" %c42_i32 : i32

    // Probe initial value
    %val1 = llhd.prb %local_var : i32
    %fmt_lit1 = sim.fmt.literal "Initial value: "
    %fmt_val1 = sim.fmt.dec %val1 signed : i32
    %fmt_nl1 = sim.fmt.literal "\0A"
    %fmt1 = sim.fmt.concat (%fmt_lit1, %fmt_val1, %fmt_nl1)
    sim.proc.print %fmt1

    // Drive a new value with epsilon delay
    llhd.drv %local_var, %c100_i32 after %time : i32

    // Probe after epsilon drive (should see 100 due to immediate read-after-write)
    %val2 = llhd.prb %local_var : i32
    %fmt_lit2 = sim.fmt.literal "After modification: "
    %fmt_val2 = sim.fmt.dec %val2 signed : i32
    %fmt_nl2 = sim.fmt.literal "\0A"
    %fmt2 = sim.fmt.concat (%fmt_lit2, %fmt_val2, %fmt_nl2)
    sim.proc.print %fmt2
  } : () -> ()
  hw.output
}

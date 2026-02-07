// RUN: circt-sim %s --top=test_runtime_sig_process --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test runtime signal creation in llhd.process blocks.
// This tests that local variables can be created dynamically during process execution.

// CHECK: [circt-sim] Found 1 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: Process local var: 42
// CHECK: [circt-sim] Simulation completed at time 0 fs
// CHECK: [circt-sim] Simulation completed

hw.module @test_runtime_sig_process() {
  llhd.process {
    %c42_i32 = hw.constant 42 : i32
    // Create a runtime signal (local variable)
    %local_var = llhd.sig name "local_var" %c42_i32 : i32
    // Probe its value
    %val = llhd.prb %local_var : i32
    // Print it
    %fmt_lit = sim.fmt.literal "Process local var: "
    %fmt_val = sim.fmt.dec %val signed : i32
    %fmt_nl = sim.fmt.literal "\0A"
    %fmt = sim.fmt.concat (%fmt_lit, %fmt_val, %fmt_nl)
    sim.proc.print %fmt
    llhd.halt
  }
  hw.output
}

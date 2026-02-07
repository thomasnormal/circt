// RUN: circt-sim %s --top=test_posedge_bit --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// Test LLHD process with posedge detection for 2-state (bit) clock signal.
// This tests the scenario where the process caches the old signal value before
// a wait and compares it with the new value after the wait to detect a posedge.
// Bug fix: getValue must check cache before re-reading probe results.

// CHECK: [circt-sim] Found 2 LLHD processes
// CHECK: [circt-sim] Registered 1 LLHD signals and 2 LLHD processes
// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed at time 5000000 fs
// CHECK: [circt-sim] Simulation completed

hw.module @test_posedge_bit() {
  %false = hw.constant false
  %true = hw.constant true
  %clk = llhd.sig %false : i1
  // Probe the clock outside both processes - this value is used for
  // sensitivity but must not override cached values from inside the process
  %clk_init = llhd.prb %clk : i1
  %delta = llhd.constant_time <0ns, 0d, 1e>
  %delay = llhd.constant_time <5000000fs, 0d, 0e>

  // Process 1: Toggle clock after delay
  llhd.process {
    cf.br ^bb1
  ^bb1:
    llhd.wait delay %delay, ^bb2
  ^bb2:
    %curr_clk = llhd.prb %clk : i1
    %next_clk = comb.xor %curr_clk, %true : i1
    llhd.drv %clk, %next_clk after %delta : i1
    cf.br ^bb1
  }

  // Process 2: Wait for posedge of clock, then terminate
  // This mimics the pattern generated for: always @(posedge clk) $finish;
  llhd.process {
    cf.br ^bb1
  ^bb1:
    // Capture the OLD clock value BEFORE waiting
    %old_clk = llhd.prb %clk : i1
    // Wait for any change on the clock signal (using module-level probe for sensitivity)
    llhd.wait (%clk_init : i1), ^bb2
  ^bb2:
    // Capture the NEW clock value AFTER waiting
    %new_clk = llhd.prb %clk : i1
    // Detect posedge: old was 0, new is 1
    %was_low = comb.xor bin %old_clk, %true : i1
    %posedge = comb.and bin %was_low, %new_clk : i1
    // If posedge detected, terminate; otherwise loop back
    cf.cond_br %posedge, ^bb3, ^bb1
  ^bb3:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}

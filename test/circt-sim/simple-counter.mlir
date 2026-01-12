// RUN: circt-sim %s --top=Counter --max-time=100000 --sim-stats 2>&1 | FileCheck %s
// REQUIRES: circt-sim

// A simple 8-bit counter design for testing the simulation driver.
// This tests basic sequential logic simulation.

// CHECK: [circt-sim] Starting simulation
// CHECK: [circt-sim] Simulation completed
// CHECK: === Simulation Statistics ===

hw.module @Counter(in %clk: !seq.clock, in %rst: i1, out count: i8) {
  %c1_i8 = hw.constant 1 : i8
  %c0_i8 = hw.constant 0 : i8

  // Register for the counter
  %count_reg = seq.compreg %next_count, %clk reset %rst, %c0_i8 : i8

  // Increment logic
  %next_count = comb.add %count_reg, %c1_i8 : i8

  hw.output %count_reg : i8
}

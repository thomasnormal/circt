// RUN: circt-sim %s --top top --max-time=100 | FileCheck %s
// CHECK: Simulation completed

// Basic test to verify VPI runtime initializes without crashing.
// This test just runs a simple combinational design and ensures the VPI
// subsystem doesn't interfere with normal simulation.

hw.module @top(in %a : i8, in %b : i8, out c : i8) {
  %sum = comb.add %a, %b : i8
  hw.output %sum : i8
}

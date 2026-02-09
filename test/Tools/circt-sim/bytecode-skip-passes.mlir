// Test that circt-sim works with bytecode input and --skip-passes.
// This is the fastest startup configuration for pre-compiled designs.
// RUN: circt-as %s -o %t.mlirbc
// RUN: circt-sim %t.mlirbc --top test --skip-passes 2>&1 | FileCheck %s

// CHECK: fast startup
// CHECK: Simulation completed

hw.module @test() {
  %fmt = sim.fmt.literal "fast startup\0A"

  llhd.process {
    sim.proc.print %fmt
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}

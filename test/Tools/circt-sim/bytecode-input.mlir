// Test that circt-sim can read MLIR bytecode format.
// RUN: circt-as %s -o %t.mlirbc
// RUN: circt-sim %t.mlirbc --top test 2>&1 | FileCheck %s

// CHECK: hello from bytecode
// CHECK: Simulation completed

hw.module @test() {
  %fmt = sim.fmt.literal "hello from bytecode\0A"

  llhd.process {
    sim.proc.print %fmt
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}

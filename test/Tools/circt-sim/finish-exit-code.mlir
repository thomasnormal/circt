// RUN: not circt-sim %s --top test 2>&1 | FileCheck %s
// Test that sim.terminate failure propagates a nonzero exit code.
// The simulator should print the exit code and _exit(1).

// CHECK: before_finish = 42
// CHECK: [circt-sim] Simulation finished with exit code 1

module {
  hw.module @test() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    llhd.process {
      cf.br ^start
    ^start:
      // Print a value to show the process ran
      %val = arith.constant 42 : i32
      %lit = sim.fmt.literal "before_finish = "
      %d = sim.fmt.dec %val signed : i32
      %nl = sim.fmt.literal "\0A"
      %fmt = sim.fmt.concat (%lit, %d, %nl)
      sim.proc.print %fmt

      // Terminate with failure (exit code 1)
      sim.terminate failure, quiet

      llhd.wait delay %t1, ^done
    ^done:
      llhd.halt
    }
    hw.output
  }
}

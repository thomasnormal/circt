// RUN: circt-sim %s | FileCheck %s
// RUN: env CIRCT_SIM_FLUSH_PROC_PRINT=1 circt-sim %s | FileCheck %s

// CHECK: print-0
// CHECK: print-1
// CHECK: print-2
// CHECK: print-last

hw.module @test() {
  %l0 = sim.fmt.literal "print-0\0A"
  %l1 = sim.fmt.literal "print-1\0A"
  %l2 = sim.fmt.literal "print-2\0A"
  %last = sim.fmt.literal "print-last\0A"

  llhd.process {
    sim.proc.print %l0
    sim.proc.print %l1
    sim.proc.print %l2
    sim.proc.print %last
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}

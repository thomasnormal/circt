// RUN: circt-sim %s 2>&1 | FileCheck %s

// Test: tight infinite loop without llhd.wait is detected and halted.
// The per-activation step limit catches processes that loop forever
// without waiting, preventing the simulator from hanging.
// No --timeout needed: the loop is caught by the 1M-step-per-activation limit.

// CHECK: Simulation completed

hw.module @test() {
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %sig = llhd.sig %false : i1

  llhd.process {
  ^bb0:
    llhd.drv %sig, %false after %eps : i1
    cf.br ^bb1
  ^bb1:
    cf.br ^bb1
  }

  hw.output
}

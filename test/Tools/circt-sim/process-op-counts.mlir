// RUN: circt-sim --mode=analyze %s | FileCheck %s

// CHECK: === Design Analysis ===
// CHECK: Modules:

hw.module @test() {
  %false = hw.constant false
  %sig = llhd.sig %false : i1
  %eps = llhd.constant_time <0ns, 0d, 1e>

  llhd.process {
    llhd.drv %sig, %false after %eps : i1
    llhd.wait delay %eps, ^bb1
  ^bb1:
    llhd.halt
  }

  hw.output
}

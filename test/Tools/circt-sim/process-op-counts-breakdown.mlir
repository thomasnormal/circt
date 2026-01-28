// RUN: circt-sim --mode=analyze %s | FileCheck %s

// CHECK: === Design Analysis ===
// CHECK: Modules:

hw.module @test() {
  %true = hw.constant true
  %false = hw.constant false

  llhd.process {
    %v0 = comb.xor %true, %false : i1
    %v1 = comb.xor %v0, %true : i1
    llhd.halt
  }

  hw.output
}

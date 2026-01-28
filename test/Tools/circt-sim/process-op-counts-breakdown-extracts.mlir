// RUN: circt-sim --mode=analyze %s | FileCheck %s

// CHECK: === Design Analysis ===
// CHECK: Modules:

hw.module @test() {
  %val = hw.constant 0 : i128

  llhd.process {
    %b0 = comb.extract %val from 0 : (i128) -> i1
    %b70 = comb.extract %val from 70 : (i128) -> i1
    %slice = comb.extract %val from 8 : (i128) -> i16
    llhd.halt
  }

  hw.output
}

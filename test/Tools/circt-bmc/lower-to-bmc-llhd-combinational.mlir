// RUN: circt-opt --externalize-registers --lower-to-bmc="top-module=comb bound=4" %s | FileCheck %s

hw.module @comb(in %a: i1) {
  %false = hw.constant false
  %0 = llhd.combinational -> i1 {
    %1 = comb.xor %a, %a : i1
    %2 = comb.or %1, %false : i1
    llhd.yield %2 : i1
  }
  hw.output
}

// CHECK: verif.bmc
// CHECK-NOT: llhd.combinational
// CHECK: comb.xor

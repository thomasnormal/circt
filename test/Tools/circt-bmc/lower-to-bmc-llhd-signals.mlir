// RUN: circt-opt --externalize-registers --lower-to-bmc="top-module=llhd bound=4" %s | FileCheck %s

hw.module @llhd(in %a: i1) {
  %t0 = llhd.constant_time <0ns, 0d, 1e>
  %sig = llhd.sig %a : i1
  %p = llhd.prb %sig : i1
  llhd.drv %sig, %p after %t0 : i1
  hw.output
}

// CHECK: verif.bmc
// CHECK-NOT: llhd.constant_time
// CHECK-NOT: llhd.sig
// CHECK-NOT: llhd.prb
// CHECK-NOT: llhd.drv

// RUN: circt-opt --externalize-registers --lower-to-bmc="top-module=llhd_drive_order bound=4" %s | FileCheck %s

hw.module @llhd_drive_order(in %a: i1) {
  %false = hw.constant false
  %t0 = llhd.constant_time <0ns, 0d, 1e>
  %sig = llhd.sig %false : i1
  llhd.drv %sig, %false after %t0 : i1
  llhd.drv %sig, %a after %t0 : i1
  %p = llhd.prb %sig : i1
  verif.assert %p : i1
  hw.output
}

// CHECK: verif.bmc
// CHECK: comb.and bin %arg1, %arg0
// CHECK: verif.assert
// CHECK-NOT: verif.assert %false

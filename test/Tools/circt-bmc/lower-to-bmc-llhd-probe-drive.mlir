// RUN: circt-opt --externalize-registers --lower-to-bmc="top-module=llhd_drive bound=4" %s | FileCheck %s

hw.module @llhd_drive(in %a: i1) {
  %false = hw.constant false
  %t0 = llhd.constant_time <0ns, 0d, 1e>
  %sig0 = llhd.sig %a : i1
  %sig1 = llhd.sig %false : i1
  %p_pre = llhd.prb %sig1 : i1
  verif.assert %p_pre : i1
  %comb = llhd.combinational -> i1 {
    %p_comb = llhd.prb %sig1 : i1
    llhd.yield %p_comb : i1
  }
  verif.assert %comb : i1
  %p0 = llhd.prb %sig0 : i1
  llhd.drv %sig1, %p0 after %t0 : i1
  %p1 = llhd.prb %sig1 : i1
  verif.assert %p1 : i1
  hw.output
}

// CHECK: verif.bmc
// CHECK: verif.assert %arg0
// CHECK: verif.assert %arg0
// CHECK: verif.assert %arg0
// CHECK-NOT: llhd.constant_time
// CHECK-NOT: llhd.sig
// CHECK-NOT: llhd.prb
// CHECK-NOT: llhd.drv

// RUN: circt-opt --externalize-registers --lower-to-bmc="top-module=llhd_multi_drive_strength_2state bound=2" %s | FileCheck %s

hw.module @llhd_multi_drive_strength_2state(in %a: i1, in %b: i1) {
  %t0 = llhd.constant_time <0ns, 0d, 1e>
  %sig0 = llhd.sig name "sig" %a : i1
  llhd.drv %sig0, %a after %t0 strength(strong, strong) : i1
  llhd.drv %sig0, %b after %t0 strength(strong, strong) : i1
  %p0 = llhd.prb %sig0 : i1
  verif.assert %p0 : i1
  hw.output
}

// CHECK-LABEL: verif.bmc
// CHECK: bmc_input_names = ["a", "b", "sig_unknown"]
// CHECK: comb.and{{.*}}%arg2
// CHECK: verif.assert

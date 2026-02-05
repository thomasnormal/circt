// RUN: circt-opt --externalize-registers --lower-to-bmc="top-module=llhd_multi_drive_enable bound=2" %s | FileCheck %s

hw.module @llhd_multi_drive_enable(in %a: !hw.struct<value: i1, unknown: i1>,
                                   in %b: !hw.struct<value: i1, unknown: i1>,
                                   in %en: i1) {
  %t0 = llhd.constant_time <0ns, 0d, 1e>
  %true = hw.constant true
  %not_en = comb.xor %en, %true : i1
  %sig0 = llhd.sig %a : !hw.struct<value: i1, unknown: i1>
  llhd.drv %sig0, %a after %t0 if %en : !hw.struct<value: i1, unknown: i1>
  llhd.drv %sig0, %b after %t0 if %not_en : !hw.struct<value: i1, unknown: i1>
  %p0 = llhd.prb %sig0 : !hw.struct<value: i1, unknown: i1>
  %unknown = hw.struct_extract %p0["unknown"] : !hw.struct<value: i1, unknown: i1>
  verif.assert %unknown : i1
  hw.output
}

// CHECK-LABEL: verif.bmc
// CHECK: comb.replicate
// CHECK: verif.assert

// RUN: circt-opt --externalize-registers --lower-to-bmc="top-module=llhd_multi_drive bound=2" %s | FileCheck %s

hw.module @llhd_multi_drive(in %a: !hw.struct<value: i1, unknown: i1>,
                            in %b: !hw.struct<value: i1, unknown: i1>) {
  %t0 = llhd.constant_time <0ns, 0d, 1e>
  %sig0 = llhd.sig %a : !hw.struct<value: i1, unknown: i1>
  llhd.drv %sig0, %a after %t0 : !hw.struct<value: i1, unknown: i1>
  llhd.drv %sig0, %b after %t0 : !hw.struct<value: i1, unknown: i1>
  %p0 = llhd.prb %sig0 : !hw.struct<value: i1, unknown: i1>
  %unknown = hw.struct_extract %p0["unknown"] : !hw.struct<value: i1, unknown: i1>
  verif.assert %unknown : i1
  hw.output
}

// CHECK-LABEL: verif.bmc
// CHECK-DAG: %[[A_VAL:.*]] = hw.struct_extract %arg0["value"]
// CHECK-DAG: %[[A_UNK:.*]] = hw.struct_extract %arg0["unknown"]
// CHECK-DAG: %[[B_VAL:.*]] = hw.struct_extract %arg1["value"]
// CHECK-DAG: %[[B_UNK:.*]] = hw.struct_extract %arg1["unknown"]
// CHECK-DAG: %[[ONES:.*]] = hw.constant true
// CHECK: %[[A_KNOWN:.*]] = comb.xor %[[A_UNK]], %[[ONES]]
// CHECK: %[[B_KNOWN:.*]] = comb.xor %[[B_UNK]], %[[ONES]]
// CHECK: %[[VAL_DIFF:.*]] = comb.xor %[[A_VAL]], %[[B_VAL]]
// CHECK: %[[KNOWN_BOTH:.*]] = comb.and %[[A_KNOWN]], %[[B_KNOWN]]
// CHECK: %[[CONFLICT:.*]] = comb.and %[[VAL_DIFF]], %[[KNOWN_BOTH]]
// CHECK: %[[UNK_OUT:.*]] = comb.or %[[A_UNK]], %[[B_UNK]], %[[CONFLICT]]
// CHECK: %[[RESOLVED:.*]] = hw.struct_create (%{{.*}}, %[[UNK_OUT]]) : !hw.struct<value: i1, unknown: i1>
// CHECK: %[[UNK_EXTRACT:.*]] = hw.struct_extract %[[RESOLVED]]["unknown"]
// CHECK: verif.assert %[[UNK_EXTRACT]]

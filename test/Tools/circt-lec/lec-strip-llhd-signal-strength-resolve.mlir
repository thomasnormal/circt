// RUN: circt-opt --strip-llhd-interface-signals %s | FileCheck %s

hw.module @sig_strength(in %a: !hw.struct<value: i1, unknown: i1>,
                        in %b: !hw.struct<value: i1, unknown: i1>,
                        out o: i1) {
  %t0 = llhd.constant_time <0ns, 0d, 1e>
  %sig = llhd.sig %a : !hw.struct<value: i1, unknown: i1>
  llhd.drv %sig, %a after %t0 strength(weak, strong) : !hw.struct<value: i1, unknown: i1>
  llhd.drv %sig, %b after %t0 strength(strong, strong) : !hw.struct<value: i1, unknown: i1>
  %p0 = llhd.prb %sig : !hw.struct<value: i1, unknown: i1>
  %unknown = hw.struct_extract %p0["unknown"] : !hw.struct<value: i1, unknown: i1>
  hw.output %unknown : i1
}

// CHECK-LABEL: hw.module @sig_strength
// CHECK-DAG: hw.struct_extract %a["value"]
// CHECK-DAG: hw.struct_extract %b["value"]
// CHECK: hw.struct_create
// CHECK: hw.output
// CHECK-NOT: llhd.drv
// CHECK-NOT: llhd.prb

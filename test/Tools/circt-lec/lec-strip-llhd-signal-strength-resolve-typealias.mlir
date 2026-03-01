// RUN: circt-opt --strip-llhd-interface-signals %s | FileCheck %s

module {
  hw.module @m(
      in %a : !hw.typealias<@foo::@FS1, !hw.struct<value: i1, unknown: i1>>,
      in %b : !hw.typealias<@foo::@FS1, !hw.struct<value: i1, unknown: i1>>,
      out out_o : !hw.typealias<@foo::@FS1, !hw.struct<value: i1, unknown: i1>>) {
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %sig = llhd.sig %a : !hw.typealias<@foo::@FS1, !hw.struct<value: i1, unknown: i1>>
    llhd.drv %sig, %a after %t0 strength(weak, strong) : !hw.typealias<@foo::@FS1, !hw.struct<value: i1, unknown: i1>>
    llhd.drv %sig, %b after %t0 strength(strong, strong) : !hw.typealias<@foo::@FS1, !hw.struct<value: i1, unknown: i1>>
    %p = llhd.prb %sig : !hw.typealias<@foo::@FS1, !hw.struct<value: i1, unknown: i1>>
    hw.output %p : !hw.typealias<@foo::@FS1, !hw.struct<value: i1, unknown: i1>>
  }
}

// Alias-backed 4-state signals should use 4-state resolution logic instead of
// falling back to 2-state unknown-input abstraction.
// CHECK-LABEL: hw.module @m(
// CHECK-SAME: in %a : !hw.typealias<@foo::@FS1, !hw.struct<value: i1, unknown: i1>>
// CHECK-SAME: in %b : !hw.typealias<@foo::@FS1, !hw.struct<value: i1, unknown: i1>>
// CHECK-NOT: sig_unknown
// CHECK-NOT: circt.bmc_abstracted_llhd_interface_inputs
// CHECK-NOT: circt.lec_abstracted_llhd_interface_inputs
// CHECK: hw.output
// CHECK-NOT: llhd.

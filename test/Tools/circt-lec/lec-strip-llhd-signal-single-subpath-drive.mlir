// RUN: circt-opt --strip-llhd-interface-signals %s | FileCheck %s

module {
  // A single drive to a subpath must not be treated as a whole-signal
  // combinational forward.
  hw.module @m(
      in %in_lock : !hw.struct<value: i1, unknown: i1>,
      out out_o : !hw.struct<lock: !hw.struct<value: i1, unknown: i1>, mode: !hw.struct<value: i2, unknown: i2>>) {
    %c0 = hw.constant 0 : i1
    %z1 = hw.constant 0 : i1
    %z2 = hw.constant 0 : i2
    %lock_init = hw.struct_create (%z1, %z1) : !hw.struct<value: i1, unknown: i1>
    %mode_init = hw.struct_create (%z2, %z2) : !hw.struct<value: i2, unknown: i2>
    %elem_init = hw.struct_create (%lock_init, %mode_init) : !hw.struct<lock: !hw.struct<value: i1, unknown: i1>, mode: !hw.struct<value: i2, unknown: i2>>
    %arr_init = hw.array_create %elem_init, %elem_init : !hw.struct<lock: !hw.struct<value: i1, unknown: i1>, mode: !hw.struct<value: i2, unknown: i2>>
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %sig = llhd.sig %arr_init : !hw.array<2xstruct<lock: !hw.struct<value: i1, unknown: i1>, mode: !hw.struct<value: i2, unknown: i2>>>
    %elem_ref = llhd.sig.array_get %sig[%c0] : <!hw.array<2xstruct<lock: !hw.struct<value: i1, unknown: i1>, mode: !hw.struct<value: i2, unknown: i2>>>>
    %lock_ref = llhd.sig.struct_extract %elem_ref["lock"] : <!hw.struct<lock: !hw.struct<value: i1, unknown: i1>, mode: !hw.struct<value: i2, unknown: i2>>>
    llhd.drv %lock_ref, %in_lock after %t0 : !hw.struct<value: i1, unknown: i1>
    %p = llhd.prb %elem_ref : !hw.struct<lock: !hw.struct<value: i1, unknown: i1>, mode: !hw.struct<value: i2, unknown: i2>>
    hw.output %p : !hw.struct<lock: !hw.struct<value: i1, unknown: i1>, mode: !hw.struct<value: i2, unknown: i2>>
  }
}

// CHECK-LABEL: hw.module @m(
// CHECK-NOT: llhd.
// CHECK: hw.output {{.*}} : !hw.struct<lock: !hw.struct<value: i1, unknown: i1>, mode: !hw.struct<value: i2, unknown: i2>>

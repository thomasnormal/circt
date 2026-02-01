// RUN: circt-lec --emit-mlir --strict-llhd -c1=top -c2=top %s %s 2>&1 | FileCheck %s

// Strict LEC should resolve conflicting unconditional drives to an X-like value.

// CHECK: smt.solver
// CHECK: smt.check

module {
  hw.module @top(out o : !hw.struct<value: i1, unknown: i1>) {
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %val0 = hw.constant 0 : i1
    %unk0 = hw.constant 0 : i1
    %val1 = hw.constant 1 : i1
    %unk1 = hw.constant 0 : i1
    %init = hw.struct_create (%val0, %unk0) : !hw.struct<value: i1, unknown: i1>
    %drv0 = hw.struct_create (%val0, %unk0) : !hw.struct<value: i1, unknown: i1>
    %drv1 = hw.struct_create (%val1, %unk1) : !hw.struct<value: i1, unknown: i1>
    %sig = llhd.sig %init : !hw.struct<value: i1, unknown: i1>
    llhd.drv %sig, %drv0 after %t0 : !hw.struct<value: i1, unknown: i1>
    llhd.drv %sig, %drv1 after %t0 : !hw.struct<value: i1, unknown: i1>
    %val = llhd.prb %sig : !hw.struct<value: i1, unknown: i1>
    hw.output %val : !hw.struct<value: i1, unknown: i1>
  }
}

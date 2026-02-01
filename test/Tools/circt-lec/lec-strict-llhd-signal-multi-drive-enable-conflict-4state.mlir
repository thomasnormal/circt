// RUN: circt-lec --emit-mlir --strict-llhd -c1=top -c2=top %s %s 2>&1 | FileCheck %s

// CHECK: smt.solver

module {
  hw.module @top(in %a : !hw.struct<value: i1, unknown: i1>,
                 in %b : !hw.struct<value: i1, unknown: i1>,
                 in %en0 : i1, in %en1 : i1,
                 out o : !hw.struct<value: i1, unknown: i1>) {
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %val0 = hw.constant 0 : i1
    %unk0 = hw.constant 0 : i1
    %init = hw.struct_create (%val0, %unk0) : !hw.struct<value: i1, unknown: i1>
    %sig = llhd.sig %init : !hw.struct<value: i1, unknown: i1>
    llhd.drv %sig, %a after %t0 if %en0 : !hw.struct<value: i1, unknown: i1>
    llhd.drv %sig, %b after %t0 if %en1 : !hw.struct<value: i1, unknown: i1>
    %p = llhd.prb %sig : !hw.struct<value: i1, unknown: i1>
    hw.output %p : !hw.struct<value: i1, unknown: i1>
  }
}

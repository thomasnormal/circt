// RUN: circt-lec --emit-mlir --lec-strict -c1=top -c2=top %s %s | FileCheck %s

// CHECK: smt.solver
// CHECK: smt.check

module {
  hw.module @top(inout %io : !hw.struct<value: i1, unknown: i1>, out o : i1) {
    %v0 = hw.constant false
    %u0 = hw.constant false
    %drv = hw.struct_create (%v0, %u0) : !hw.struct<value: i1, unknown: i1>
    sv.assign %io, %drv : !hw.struct<value: i1, unknown: i1>
    %read = sv.read_inout %io : !hw.inout<!hw.struct<value: i1, unknown: i1>>
    %unknown = hw.struct_extract %read["unknown"] : !hw.struct<value: i1, unknown: i1>
    hw.output %unknown : i1
  }
}

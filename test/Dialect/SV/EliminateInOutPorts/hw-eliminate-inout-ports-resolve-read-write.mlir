// RUN: circt-opt --hw-eliminate-inout-ports=resolve-read-write %s | FileCheck %s

// CHECK: hw.module @top
// CHECK-SAME: in %io_rd : !hw.struct<value: i1, unknown: i1>
// CHECK-SAME: out io_wr : !hw.struct<value: i1, unknown: i1>
// CHECK-NOT: sv.read_inout
// CHECK-NOT: sv.assign
// CHECK: comb.or

module {
  hw.module @top(inout %io : !hw.struct<value: i1, unknown: i1>, out o : i1) {
    %val0 = hw.constant 0 : i1
    %unk0 = hw.constant 0 : i1
    %drv = hw.struct_create (%val0, %unk0) : !hw.struct<value: i1, unknown: i1>
    sv.assign %io, %drv : !hw.struct<value: i1, unknown: i1>
    %read = sv.read_inout %io : !hw.inout<!hw.struct<value: i1, unknown: i1>>
    %unknown = hw.struct_extract %read["unknown"] : !hw.struct<value: i1, unknown: i1>
    hw.output %unknown : i1
  }
}

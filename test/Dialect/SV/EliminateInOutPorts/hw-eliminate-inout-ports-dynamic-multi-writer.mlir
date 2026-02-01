// RUN: circt-opt --hw-eliminate-inout-ports="resolve-read-write" %s | FileCheck %s

// CHECK-LABEL: hw.module @top
// CHECK-SAME: in %io_rd_dyn0 : !hw.array<4xstruct<value: i1, unknown: i1>>
// CHECK-SAME: out io_wr_dyn0 : !hw.array<4xstruct<value: i1, unknown: i1>>
// CHECK-NOT: io_rd_dyn1
// CHECK-NOT: io_wr_dyn1
// CHECK-DAG: hw.array_get {{.*}}[%sel0]
// CHECK-DAG: hw.array_get {{.*}}[%sel1]
// CHECK: hw.array_inject

module {
  hw.module @top(
      inout %io : !hw.array<4x!hw.struct<value: i1, unknown: i1>>,
      in %sel0 : i2, in %sel1 : i2, out o : i1) {
    %v0 = hw.constant 0 : i1
    %u0 = hw.constant 0 : i1
    %v1 = hw.constant 1 : i1
    %u1 = hw.constant 0 : i1
    %drv0 = hw.struct_create (%v0, %u0) : !hw.struct<value: i1, unknown: i1>
    %drv1 = hw.struct_create (%v1, %u1) : !hw.struct<value: i1, unknown: i1>
    %elem0 = sv.array_index_inout %io[%sel0] : !hw.inout<array<4xstruct<value: i1, unknown: i1>>>, i2
    sv.assign %elem0, %drv0 : !hw.struct<value: i1, unknown: i1>
    %elem1 = sv.array_index_inout %io[%sel1] : !hw.inout<array<4xstruct<value: i1, unknown: i1>>>, i2
    sv.assign %elem1, %drv1 : !hw.struct<value: i1, unknown: i1>
    hw.output %v0 : i1
  }
}

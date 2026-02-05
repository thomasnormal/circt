// RUN: circt-opt --hw-eliminate-inout-ports %s | FileCheck %s

// CHECK-LABEL: hw.module @top
// CHECK-SAME: in %io_rd_idx2 : i1
// CHECK-SAME: out io_wr_idx1 : i1
// CHECK-NOT: sv.array_index_inout
// CHECK-NOT: sv.read_inout
// CHECK-NOT: sv.assign

module {
  hw.module @top(inout %io : !hw.array<4xi1>, out o : i1) {
    %idx2 = hw.constant 2 : i2
    %idx1 = hw.constant 1 : i2
    %elem2 = sv.array_index_inout %io[%idx2] : !hw.inout<array<4xi1>>, i2
    %read = sv.read_inout %elem2 : !hw.inout<i1>
    %elem1 = sv.array_index_inout %io[%idx1] : !hw.inout<array<4xi1>>, i2
    sv.assign %elem1, %read : i1
    hw.output %read : i1
  }

  // CHECK-LABEL: hw.module @top_dynamic
  // CHECK-SAME: in %io_rd_dyn0 : !hw.array<4xi1>
  // CHECK-SAME: out io_wr_dyn0 : !hw.array<4xi1>
  // CHECK-NOT: sv.array_index_inout
  // CHECK-NOT: sv.read_inout
  // CHECK-NOT: sv.assign
  // CHECK: %[[READ:.*]] = hw.array_get %io_rd_dyn0[%sel] : !hw.array<4xi1>, i2
  // CHECK: %[[WRITE:.*]] = hw.array_inject %io_rd_dyn0[%sel], %{{.*}} : !hw.array<4xi1>, i2
  // CHECK: hw.output %[[WRITE]], %[[READ]] : !hw.array<4xi1>, i1
  hw.module @top_dynamic(inout %io : !hw.array<4xi1>, in %sel : i2, out o : i1) {
    %c1_i1 = hw.constant 1 : i1
    %elem = sv.array_index_inout %io[%sel] : !hw.inout<array<4xi1>>, i2
    %read = sv.read_inout %elem : !hw.inout<i1>
    sv.assign %elem, %c1_i1 : i1
    hw.output %read : i1
  }

  // CHECK-LABEL: hw.module @top_dynamic_field
  // CHECK-SAME: in %io_rd_bar_dyn0 : !hw.array<4xstruct<foo: i1, bar: i1>>
  // CHECK-SAME: out io_wr_dyn0 : !hw.array<4xstruct<foo: i1, bar: i1>>
  // CHECK-NOT: sv.array_index_inout
  // CHECK-NOT: sv.struct_field_inout
  // CHECK-NOT: sv.read_inout
  // CHECK-NOT: sv.assign
  // CHECK: hw.array_get %io_rd_bar_dyn0[%sel] : !hw.array<4xstruct<foo: i1, bar: i1>>, i2
  // CHECK: hw.struct_extract {{.*}}["bar"] : !hw.struct<foo: i1, bar: i1>
  // CHECK: hw.struct_inject {{.*}}["bar"], {{.*}} : !hw.struct<foo: i1, bar: i1>
  // CHECK: hw.array_inject %io_rd_bar_dyn0[%sel], {{.*}} : !hw.array<4xstruct<foo: i1, bar: i1>>, i2
  // CHECK: hw.output {{.*}}, {{.*}} : !hw.array<4xstruct<foo: i1, bar: i1>>, i1
  hw.module @top_dynamic_field(
      inout %io : !hw.array<4x!hw.struct<foo: i1, bar: i1>>, in %sel : i2,
      out o : i1) {
    %elem = sv.array_index_inout %io[%sel] : !hw.inout<array<4xstruct<foo: i1, bar: i1>>>, i2
    %field = sv.struct_field_inout %elem["bar"] : !hw.inout<struct<foo: i1, bar: i1>>
    %read = sv.read_inout %field : !hw.inout<i1>
    sv.assign %field, %read : i1
    hw.output %read : i1
  }

  // CHECK-LABEL: hw.module @top_dynamic_nested
  // CHECK-SAME: in %io_rd_dyn0 : !hw.array<4xarray<2xi1>>
  // CHECK-SAME: out io_wr_dyn0 : !hw.array<4xarray<2xi1>>
  // CHECK-NOT: sv.array_index_inout
  // CHECK-NOT: sv.read_inout
  // CHECK-NOT: sv.assign
  // CHECK: hw.array_get %io_rd_dyn0[%sel0] : !hw.array<4xarray<2xi1>>, i2
  // CHECK: hw.array_get {{.*}}[%sel1] : !hw.array<2xi1>, i1
  // CHECK: hw.array_inject {{.*}}[%sel1], {{.*}} : !hw.array<2xi1>, i1
  // CHECK: hw.array_inject %io_rd_dyn0[%sel0], {{.*}} : !hw.array<4xarray<2xi1>>, i2
  // CHECK: hw.output {{.*}}, {{.*}} : !hw.array<4xarray<2xi1>>, i1
  hw.module @top_dynamic_nested(
      inout %io : !hw.array<4x!hw.array<2xi1>>, in %sel0 : i2, in %sel1 : i1,
      out o : i1) {
    %elem0 = sv.array_index_inout %io[%sel0] : !hw.inout<array<4xarray<2xi1>>>, i2
    %elem1 = sv.array_index_inout %elem0[%sel1] : !hw.inout<array<2xi1>>, i1
    %read = sv.read_inout %elem1 : !hw.inout<i1>
    sv.assign %elem1, %read : i1
    hw.output %read : i1
  }
}

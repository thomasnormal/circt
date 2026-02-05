// RUN: circt-opt --hw-eliminate-inout-ports %s | FileCheck %s

// CHECK-LABEL: hw.module @top
// CHECK-SAME: in %io_rd_a : i1
// CHECK-SAME: out io_wr_b : i1
// CHECK-NOT: sv.struct_field_inout
// CHECK-NOT: sv.read_inout
// CHECK-NOT: sv.assign

module {
  hw.module @top(inout %io : !hw.struct<a: i1, b: i1>, out o : i1) {
    %field_a = sv.struct_field_inout %io["a"] : !hw.inout<struct<a: i1, b: i1>>
    %read = sv.read_inout %field_a : !hw.inout<i1>
    %c1 = hw.constant 1 : i1
    %field_b = sv.struct_field_inout %io["b"] : !hw.inout<struct<a: i1, b: i1>>
    sv.assign %field_b, %c1 : i1
    hw.output %read : i1
  }
}

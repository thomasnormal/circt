// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// CHECK-LABEL: func.func @FourStateAddPartialUnknown
// CHECK: hw.struct_create
// CHECK-SAME: !hw.struct<value: i4, unknown: i4>
// CHECK: return {{.*}} : !hw.struct<value: i4, unknown: i4>
func.func @FourStateAddPartialUnknown() -> !moore.l4 {
  %a = moore.constant b000X : l4
  %b = moore.constant b0010 : l4
  %x = moore.add %a, %b : l4
  return %x : !moore.l4
}

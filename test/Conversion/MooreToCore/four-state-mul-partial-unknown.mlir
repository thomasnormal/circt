// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// CHECK-LABEL: func.func @FourStateMulPartialUnknown
// CHECK: hw.struct_create
// CHECK-SAME: !hw.struct<value: i4, unknown: i4>
// CHECK: return {{.*}} : !hw.struct<value: i4, unknown: i4>
func.func @FourStateMulPartialUnknown() -> !moore.l4 {
  %a = moore.constant b0001 : l4
  %b = moore.constant bX000 : l4
  %x = moore.mul %a, %b : l4
  return %x : !moore.l4
}

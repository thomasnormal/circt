// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// CHECK-LABEL: func.func @FourStateXorSelf
// CHECK: hw.struct_create
// CHECK-SAME: !hw.struct<value: i4, unknown: i4>
func.func @FourStateXorSelf(%a: !moore.l4) -> !moore.l4 {
  %x = moore.xor %a, %a : l4
  return %x : !moore.l4
}

// CHECK-LABEL: func.func @FourStateXorNot
// CHECK: hw.struct_create
// CHECK-SAME: !hw.struct<value: i4, unknown: i4>
func.func @FourStateXorNot(%a: !moore.l4) -> !moore.l4 {
  %b = moore.not %a : l4
  %x = moore.xor %a, %b : l4
  return %x : !moore.l4
}

// CHECK-LABEL: func.func @FourStateAndNot
// CHECK: hw.struct_create
// CHECK-SAME: !hw.struct<value: i4, unknown: i4>
func.func @FourStateAndNot(%a: !moore.l4) -> !moore.l4 {
  %b = moore.not %a : l4
  %x = moore.and %a, %b : l4
  return %x : !moore.l4
}

// CHECK-LABEL: func.func @FourStateOrNot
// CHECK: hw.struct_create
// CHECK-SAME: !hw.struct<value: i4, unknown: i4>
func.func @FourStateOrNot(%a: !moore.l4) -> !moore.l4 {
  %b = moore.not %a : l4
  %x = moore.or %a, %b : l4
  return %x : !moore.l4
}

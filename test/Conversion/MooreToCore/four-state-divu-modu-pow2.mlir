// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// CHECK-LABEL: func.func @FourStateDivuByFour
// CHECK-NOT: comb.divu
// CHECK: comb.shru
// CHECK: comb.shru
func.func @FourStateDivuByFour(%a: !moore.l8) -> !moore.l8 {
  %c = moore.constant b00000100 : l8
  %x = moore.divu %a, %c : l8
  return %x : !moore.l8
}

// CHECK-LABEL: func.func @FourStateModuByFour
// CHECK-NOT: comb.modu
// CHECK: comb.and
// CHECK: comb.and
func.func @FourStateModuByFour(%a: !moore.l8) -> !moore.l8 {
  %c = moore.constant b00000100 : l8
  %x = moore.modu %a, %c : l8
  return %x : !moore.l8
}

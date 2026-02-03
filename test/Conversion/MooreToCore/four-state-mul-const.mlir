// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// CHECK-LABEL: func.func @FourStateMulByOne
// CHECK-NOT: comb.mul
// CHECK: return %arg0
func.func @FourStateMulByOne(%a: !moore.l4) -> !moore.l4 {
  %b = moore.constant b0001 : l4
  %x = moore.mul %a, %b : l4
  return %x : !moore.l4
}

// CHECK-LABEL: func.func @FourStateMulByZero
// CHECK: hw.aggregate_constant [0 : i4, 0 : i4]
// CHECK-NOT: comb.mul
func.func @FourStateMulByZero(%a: !moore.l4) -> !moore.l4 {
  %b = moore.constant b0000 : l4
  %x = moore.mul %a, %b : l4
  return %x : !moore.l4
}

// CHECK-LABEL: func.func @FourStateMulByTwo
// CHECK-NOT: comb.mul
// CHECK: comb.shl
// CHECK: comb.shl
func.func @FourStateMulByTwo(%a: !moore.l4) -> !moore.l4 {
  %b = moore.constant b0010 : l4
  %x = moore.mul %a, %b : l4
  return %x : !moore.l4
}

// CHECK-LABEL: func.func @FourStateMulByThree
// CHECK-NOT: comb.mul
// CHECK: comb.shl
// CHECK: comb.shl
func.func @FourStateMulByThree(%a: !moore.l4) -> !moore.l4 {
  %b = moore.constant b0011 : l4
  %x = moore.mul %a, %b : l4
  return %x : !moore.l4
}

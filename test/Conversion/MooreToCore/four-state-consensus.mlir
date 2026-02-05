// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// CHECK-LABEL: func.func @FourStateConsensusXor
// CHECK: return %arg0 : !hw.struct<value: i4, unknown: i4>
func.func @FourStateConsensusXor(%a: !moore.l4, %b: !moore.l4) -> !moore.l4 {
  %nb = moore.not %b : l4
  %and0 = moore.and %a, %b : l4
  %and1 = moore.and %a, %nb : l4
  %x = moore.xor %and0, %and1 : l4
  return %x : !moore.l4
}

// CHECK-LABEL: func.func @FourStateConsensusOr
// CHECK: return %arg0 : !hw.struct<value: i4, unknown: i4>
func.func @FourStateConsensusOr(%a: !moore.l4, %b: !moore.l4) -> !moore.l4 {
  %nb = moore.not %b : l4
  %and0 = moore.and %a, %b : l4
  %and1 = moore.and %a, %nb : l4
  %x = moore.or %and0, %and1 : l4
  return %x : !moore.l4
}

// CHECK-LABEL: func.func @FourStateConsensusAnd
// CHECK: return %arg0 : !hw.struct<value: i4, unknown: i4>
func.func @FourStateConsensusAnd(%a: !moore.l4, %b: !moore.l4) -> !moore.l4 {
  %nb = moore.not %b : l4
  %or0 = moore.or %a, %b : l4
  %or1 = moore.or %a, %nb : l4
  %x = moore.and %or0, %or1 : l4
  return %x : !moore.l4
}

// CHECK-LABEL: func.func @FourStateAbsorbOr
// CHECK: return %arg0 : !hw.struct<value: i4, unknown: i4>
func.func @FourStateAbsorbOr(%a: !moore.l4, %b: !moore.l4) -> !moore.l4 {
  %and0 = moore.and %a, %b : l4
  %x = moore.or %a, %and0 : l4
  return %x : !moore.l4
}

// CHECK-LABEL: func.func @FourStateAbsorbAnd
// CHECK: return %arg0 : !hw.struct<value: i4, unknown: i4>
func.func @FourStateAbsorbAnd(%a: !moore.l4, %b: !moore.l4) -> !moore.l4 {
  %or0 = moore.or %a, %b : l4
  %x = moore.and %a, %or0 : l4
  return %x : !moore.l4
}

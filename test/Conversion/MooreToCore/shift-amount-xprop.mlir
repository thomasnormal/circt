// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// CHECK-LABEL: func.func @ShlAmountXProp
// CHECK: hw.struct_extract %arg1["unknown"]
// CHECK: [[COND:%.*]] = comb.icmp ne {{.*}}
// CHECK: comb.shl
// CHECK: comb.mux [[COND]], {{.*}}, {{.*}} : i8
// CHECK: comb.mux [[COND]], {{.*}}, {{.*}} : i8
func.func @ShlAmountXProp(%val: !moore.l8, %amt: !moore.l4) -> !moore.l8 {
  %0 = moore.shl %val, %amt : !moore.l8, !moore.l4
  return %0 : !moore.l8
}

// CHECK-LABEL: func.func @ShrAmountXProp
// CHECK: hw.struct_extract %arg1["unknown"]
// CHECK: [[COND:%.*]] = comb.icmp ne {{.*}}
// CHECK: comb.shru
// CHECK: comb.mux [[COND]], {{.*}}, {{.*}} : i8
// CHECK: comb.mux [[COND]], {{.*}}, {{.*}} : i8
func.func @ShrAmountXProp(%val: !moore.l8, %amt: !moore.l4) -> !moore.l8 {
  %0 = moore.shr %val, %amt : !moore.l8, !moore.l4
  return %0 : !moore.l8
}

// CHECK-LABEL: func.func @AShrAmountXProp
// CHECK: hw.struct_extract %arg1["unknown"]
// CHECK: [[COND:%.*]] = comb.icmp ne {{.*}}
// CHECK: comb.shrs
// CHECK: comb.shru
// CHECK: comb.mux [[COND]], {{.*}}, {{.*}} : i8
// CHECK: comb.mux [[COND]], {{.*}}, {{.*}} : i8
func.func @AShrAmountXProp(%val: !moore.l8, %amt: !moore.l4) -> !moore.l8 {
  %0 = moore.ashr %val, %amt : !moore.l8, !moore.l4
  return %0 : !moore.l8
}

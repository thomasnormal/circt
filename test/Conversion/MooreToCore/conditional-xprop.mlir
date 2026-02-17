// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// CHECK-LABEL: func.func @ConditionalXProp
// CHECK: hw.struct_extract %arg0["unknown"]
// CHECK: [[CONDUNK:%.*]] = comb.icmp ne {{.*}}
// CHECK: [[SELVAL:%.*]] = comb.mux {{.*}} : i4
// CHECK: [[SELUNK:%.*]] = comb.mux {{.*}} : i4
// CHECK: [[MERGEVAL:%.*]] = comb.and {{.*}} : i4
// CHECK: [[DIFF:%.*]] = comb.xor {{.*}} : i4
// CHECK: [[UNKOR:%.*]] = comb.or {{.*}} : i4
// CHECK: [[MERGEUNK:%.*]] = comb.or [[UNKOR]], [[DIFF]] : i4
// CHECK: [[FINALVAL:%.*]] = comb.mux [[CONDUNK]], [[MERGEVAL]], [[SELVAL]] : i4
// CHECK: [[FINALUNK:%.*]] = comb.mux [[CONDUNK]], [[MERGEUNK]], [[SELUNK]] : i4
// CHECK: [[ONES:%.*]] = hw.constant -1 : i4
// CHECK: [[KNOWN:%.*]] = comb.xor [[FINALUNK]], [[ONES]] : i4
// CHECK: [[MASKED:%.*]] = comb.and [[FINALVAL]], [[KNOWN]] : i4
// CHECK: hw.struct_create ([[MASKED]], [[FINALUNK]])
func.func @ConditionalXProp(%cond: !moore.l1, %a: !moore.l4, %b: !moore.l4) -> !moore.l4 {
  %0 = moore.conditional %cond : l1 -> l4 {
    moore.yield %a : l4
  } {
    moore.yield %b : l4
  }
  return %0 : !moore.l4
}

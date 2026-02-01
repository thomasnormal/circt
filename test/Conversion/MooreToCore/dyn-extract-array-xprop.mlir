// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// CHECK-LABEL: func.func @DynExtractArrayXProp
// CHECK: hw.struct_extract %arg1["unknown"]
// CHECK: [[UNKCOND:%.*]] = comb.icmp ne {{.*}}
// CHECK: comb.icmp ugt
// CHECK: [[COND:%.*]] = comb.or {{.*}}[[UNKCOND]]
// CHECK: comb.mux [[COND]]{{.*}} : i4
// CHECK: comb.mux [[COND]]{{.*}} : i4
// CHECK: hw.struct_create
func.func @DynExtractArrayXProp(%arr: !moore.array<4 x l4>, %idx: !moore.l2) -> !moore.l4 {
  %0 = moore.dyn_extract %arr from %idx : !moore.array<4 x l4>, !moore.l2 -> !moore.l4
  return %0 : !moore.l4
}

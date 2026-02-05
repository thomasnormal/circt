// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// CHECK-LABEL: func.func @DynExtractArrayXProp
// CHECK: hw.struct_extract %arg1["unknown"]
// CHECK: [[UNKCOND:%.*]] = comb.icmp ne {{.*}}
// CHECK: comb.icmp ugt
// CHECK: hw.struct_create
// CHECK: comb.mux [[UNKCOND]]{{.*}} : !hw.struct<value: i4, unknown: i4>
func.func @DynExtractArrayXProp(%arr: !moore.array<4 x l4>, %idx: !moore.l2) -> !moore.l4 {
  %0 = moore.dyn_extract %arr from %idx : !moore.array<4 x l4>, !moore.l2 -> !moore.l4
  return %0 : !moore.l4
}

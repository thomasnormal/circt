// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// CHECK-LABEL: func.func @DynExtractArraySliceXProp
// CHECK: hw.struct_extract %arg1["unknown"]
// CHECK: [[UNKCOND:%.*]] = comb.icmp ne {{.*}}
// CHECK: hw.array_slice
// CHECK: comb.mux [[UNKCOND]]
// CHECK: hw.array_create
func.func @DynExtractArraySliceXProp(%arr: !moore.array<4 x l4>, %idx: !moore.l2) -> !moore.array<2 x l4> {
  %0 = moore.dyn_extract %arr from %idx : !moore.array<4 x l4>, !moore.l2 -> !moore.array<2 x l4>
  return %0 : !moore.array<2 x l4>
}


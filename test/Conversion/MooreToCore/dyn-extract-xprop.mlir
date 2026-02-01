// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// CHECK-LABEL: func.func @DynExtractXProp
// CHECK: hw.struct_extract %arg1["unknown"]
// CHECK: [[UNKCOND:%.*]] = comb.icmp ne {{.*}}
// CHECK: comb.icmp ugt
// CHECK: [[OOB:%.*]] = comb.or {{.*}}[[UNKCOND]]
// CHECK: comb.mux [[OOB]]{{.*}} : i4
// CHECK: comb.mux [[OOB]]{{.*}} : i4
// CHECK: hw.struct_create
func.func @DynExtractXProp(%in: !moore.l8, %idx: !moore.l5) -> !moore.l4 {
  %0 = moore.dyn_extract %in from %idx : !moore.l8, !moore.l5 -> !moore.l4
  return %0 : !moore.l4
}

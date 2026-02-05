// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// CHECK-LABEL: func.func @DynExtractXProp
// CHECK: hw.struct_extract %arg1["unknown"]
// CHECK: [[UNKCOND:%.*]] = comb.icmp ne {{.*}}
// CHECK-DAG: comb.replicate
// CHECK-DAG: comb.icmp eq
// CHECK-DAG: comb.icmp ugt
// CHECK-DAG: hw.struct_create
// CHECK: comb.mux [[UNKCOND]]
func.func @DynExtractXProp(%in: !moore.l8, %idx: !moore.l5) -> !moore.l4 {
  %0 = moore.dyn_extract %in from %idx : !moore.l8, !moore.l5 -> !moore.l4
  return %0 : !moore.l4
}

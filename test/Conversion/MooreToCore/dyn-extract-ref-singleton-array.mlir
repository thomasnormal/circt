// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @DynExtractRefSingletonArrayElement
// CHECK: llhd.sig.array_get
func.func @DynExtractRefSingletonArrayElement(%j: !moore.ref<array<1 x array<1 x l3>>>, %idx: !moore.l1) -> (!moore.ref<array<1 x l3>>) {
  %0 = moore.dyn_extract_ref %j from %idx : !moore.ref<array<1 x array<1 x l3>>>, !moore.l1 -> !moore.ref<array<1 x l3>>
  return %0 : !moore.ref<array<1 x l3>>
}

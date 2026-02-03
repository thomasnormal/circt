// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// CHECK-LABEL: func.func @ConstArrayUnknownIndex
// CHECK-DAG: hw.constant -6 : i4
// CHECK-DAG: hw.constant -4 : i4
// CHECK-NOT: hw.constant -1 : i2
// CHECK: hw.struct_create
func.func @ConstArrayUnknownIndex(%idx: !moore.l2) -> !moore.l2 {
  %c0 = moore.constant b01 : l2
  %c1 = moore.constant b00 : l2
  %arr = moore.array_create %c0, %c1, %c0, %c1
    : !moore.l2, !moore.l2, !moore.l2, !moore.l2 -> !moore.array<4 x l2>
  %val = moore.dyn_extract %arr from %idx : !moore.array<4 x l2>, !moore.l2 -> !moore.l2
  return %val : !moore.l2
}

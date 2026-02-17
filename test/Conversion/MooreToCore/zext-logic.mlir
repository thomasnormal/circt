// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

module {
  // CHECK-LABEL: func.func @LogicZExt
  // CHECK-SAME: (%arg0: !hw.struct<value: i1, unknown: i1>) -> !hw.struct<value: i4, unknown: i4>
  // CHECK: %[[VAL:.*]] = hw.struct_extract %arg0["value"]
  // CHECK: %[[UNK:.*]] = hw.struct_extract %arg0["unknown"]
  // CHECK: %[[VEXT:.*]] = comb.concat {{%.+}}, %[[VAL]] : i3, i1
  // CHECK: %[[UEXT:.*]] = comb.concat {{%.+}}, %[[UNK]] : i3, i1
  // CHECK: %[[OUT:.*]] = hw.struct_create (%[[VEXT]], %[[UEXT]]) : !hw.struct<value: i4, unknown: i4>
  // CHECK: return %[[OUT]]
  func.func @LogicZExt(%arg0: !moore.l1) -> !moore.l4 {
    %0 = moore.zext %arg0 : l1 -> l4
    return %0 : !moore.l4
  }
}

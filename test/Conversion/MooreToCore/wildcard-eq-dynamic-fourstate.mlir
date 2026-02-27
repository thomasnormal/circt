// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Regression: for wildcard equality, unknown LHS bits only taint the result
// when those bits are not masked by wildcard (X/Z) bits on the RHS.

module {
  // CHECK-LABEL: func.func @WildcardEqDynamic
  // CHECK-SAME: (%[[LHS:.*]]: !hw.struct<value: i2, unknown: i2>, %[[RHS:.*]]: !hw.struct<value: i2, unknown: i2>) -> !hw.struct<value: i1, unknown: i1>
  func.func @WildcardEqDynamic(%lhs: !moore.l2, %rhs: !moore.l2) -> !moore.l1 {
    // CHECK: %[[LV:.*]] = hw.struct_extract %[[LHS]]["value"]
    // CHECK: %[[LU:.*]] = hw.struct_extract %[[LHS]]["unknown"]
    // CHECK: %[[RV:.*]] = hw.struct_extract %[[RHS]]["value"]
    // CHECK: %[[RU:.*]] = hw.struct_extract %[[RHS]]["unknown"]
    // CHECK: %[[ONES:.*]] = hw.constant -1 : i2
    // CHECK: %[[MASK:.*]] = comb.xor %[[RU]], %[[ONES]] : i2
    // CHECK: %[[LUM:.*]] = comb.and %[[LU]], %[[MASK]] : i2
    // CHECK: %[[ZEROU:.*]] = hw.constant 0 : i2
    // CHECK: %[[HASUNK:.*]] = comb.icmp ne %[[LUM]], %[[ZEROU]] : i2
    // CHECK: %[[NOTLUM:.*]] = comb.xor %[[LUM]], %[[ONES]] : i2
    // CHECK: %[[KNOWNMASK:.*]] = comb.and %[[MASK]], %[[NOTLUM]] : i2
    // CHECK: %[[KLV:.*]] = comb.and %[[LV]], %[[KNOWNMASK]] : i2
    // CHECK: %[[KRV:.*]] = comb.and %[[RV]], %[[KNOWNMASK]] : i2
    // CHECK: %[[KNOWNEQ:.*]] = comb.icmp eq %[[KLV]], %[[KRV]] : i2
    // CHECK: %[[ONE:.*]] = hw.constant true
    // CHECK: %[[NOUNK:.*]] = comb.xor %[[HASUNK]], %[[ONE]] : i1
    // CHECK: %[[UNKRES:.*]] = comb.and %[[HASUNK]], %[[KNOWNEQ]] : i1
    // CHECK: %[[RESV:.*]] = comb.and %[[KNOWNEQ]], %[[NOUNK]] : i1
    // CHECK: %[[RES:.*]] = hw.struct_create (%[[RESV]], %[[UNKRES]]) : !hw.struct<value: i1, unknown: i1>
    // CHECK: return %[[RES]] : !hw.struct<value: i1, unknown: i1>
    %eq = moore.wildcard_eq %lhs, %rhs : l2 -> l1
    return %eq : !moore.l1
  }
}

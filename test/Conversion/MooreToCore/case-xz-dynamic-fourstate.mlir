// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Regression: dynamic casez/casex comparisons on 4-state values must honor
// unknown bits from non-constant operands, not just constants.
//
// casez:
// - ignore only Z bits (value & unknown)
// - compare both value and unknown masks on remaining bits
//
// casex:
// - ignore all unknown bits (X/Z)
// - compare value on remaining bits

module {
  // CHECK-LABEL: func.func @CaseXZDynamic
  // CHECK-SAME: (%[[LHS:.*]]: !hw.struct<value: i4, unknown: i4>, %[[RHS:.*]]: !hw.struct<value: i4, unknown: i4>) -> (i1, i1)
  func.func @CaseXZDynamic(%lhs: !moore.l4, %rhs: !moore.l4) -> (!moore.i1, !moore.i1) {
    // CHECK: %[[LV:.*]] = hw.struct_extract %[[LHS]]["value"]
    // CHECK: %[[LU:.*]] = hw.struct_extract %[[LHS]]["unknown"]
    // CHECK: %[[RV:.*]] = hw.struct_extract %[[RHS]]["value"]
    // CHECK: %[[RU:.*]] = hw.struct_extract %[[RHS]]["unknown"]
    // CHECK: %[[LZ:.*]] = comb.and %[[LV]], %[[LU]] : i4
    // CHECK: %[[RZ:.*]] = comb.and %[[RV]], %[[RU]] : i4
    // CHECK: %[[IGNZ:.*]] = comb.or %[[LZ]], %[[RZ]] : i4
    // CHECK: %[[ONESZ:.*]] = hw.constant -1 : i4
    // CHECK: %[[MASKZ:.*]] = comb.xor %[[IGNZ]], %[[ONESZ]] : i4
    // CHECK: %[[LMV:.*]] = comb.and %[[LV]], %[[MASKZ]] : i4
    // CHECK: %[[RMV:.*]] = comb.and %[[RV]], %[[MASKZ]] : i4
    // CHECK: %[[VEQ:.*]] = comb.icmp ceq %[[LMV]], %[[RMV]] : i4
    // CHECK: %[[LMU:.*]] = comb.and %[[LU]], %[[MASKZ]] : i4
    // CHECK: %[[RMU:.*]] = comb.and %[[RU]], %[[MASKZ]] : i4
    // CHECK: %[[UEQ:.*]] = comb.icmp ceq %[[LMU]], %[[RMU]] : i4
    // CHECK: %[[CASEZ:.*]] = comb.and %[[VEQ]], %[[UEQ]] : i1
    %cz = moore.casez_eq %lhs, %rhs : l4

    // CHECK: %[[LV2:.*]] = hw.struct_extract %[[LHS]]["value"]
    // CHECK: %[[LU2:.*]] = hw.struct_extract %[[LHS]]["unknown"]
    // CHECK: %[[RV2:.*]] = hw.struct_extract %[[RHS]]["value"]
    // CHECK: %[[RU2:.*]] = hw.struct_extract %[[RHS]]["unknown"]
    // CHECK: %[[IGNX:.*]] = comb.or %[[LU2]], %[[RU2]] : i4
    // CHECK: %[[ONESX:.*]] = hw.constant -1 : i4
    // CHECK: %[[MASKX:.*]] = comb.xor %[[IGNX]], %[[ONESX]] : i4
    // CHECK: %[[LMVX:.*]] = comb.and %[[LV2]], %[[MASKX]] : i4
    // CHECK: %[[RMVX:.*]] = comb.and %[[RV2]], %[[MASKX]] : i4
    // CHECK: %[[CASEX:.*]] = comb.icmp ceq %[[LMVX]], %[[RMVX]] : i4
    %cx = moore.casexz_eq %lhs, %rhs : l4

    // CHECK: return %[[CASEZ]], %[[CASEX]] : i1, i1
    return %cz, %cx : !moore.i1, !moore.i1
  }
}

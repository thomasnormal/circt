// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test 2-state logic operations

// CHECK-LABEL: func @TwoStateLogicOps
// CHECK-SAME: (%arg0: i4, %arg1: i4) -> (i4, i4, i4, i4)
func.func @TwoStateLogicOps(%a: !moore.i4, %b: !moore.i4) -> (!moore.i4, !moore.i4, !moore.i4, !moore.i4) {
  // CHECK-DAG: %[[ONES:.*]] = hw.constant -1 : i4
  // AND operation
  // CHECK-DAG: %[[AND:.*]] = comb.and %arg0, %arg1 : i4
  %and = moore.and %a, %b : i4

  // OR operation
  // CHECK-DAG: %[[OR:.*]] = comb.or %arg0, %arg1 : i4
  %or = moore.or %a, %b : i4

  // XOR operation
  // CHECK-DAG: %[[XOR:.*]] = comb.xor %arg0, %arg1 : i4
  %xor = moore.xor %a, %b : i4

  // NOT operation
  // CHECK-DAG: %[[NOT:.*]] = comb.xor %arg0, %[[ONES]] : i4
  %not = moore.not %a : i4

  // CHECK: return %[[AND]], %[[OR]], %[[XOR]], %[[NOT]]
  return %and, %or, %xor, %not : !moore.i4, !moore.i4, !moore.i4, !moore.i4
}

// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test 4-state constants with X/Z bits

// CHECK-LABEL: func @FourStateConstants
func.func @FourStateConstants() {
  // Four-valued constant with all 0s - should become plain 0
  // CHECK: hw.constant 0 : i4
  %c0 = moore.constant 0 : !moore.l4

  // Four-valued constant with all 1s - should become plain 15
  // CHECK: hw.constant -1 : i4
  %c1 = moore.constant 15 : !moore.l4

  // Four-valued constant with X bits - X maps to 0
  // b1X00 = value 1000, unknown 0100 -> APInt with unknown mapped to 0 = 1000 = 8 (signed: -8)
  // CHECK: hw.constant -8 : i4
  %cx = moore.constant b1X00 : l4

  // Four-valued constant with Z bits - Z maps to 0
  // b10Z0 = value 1010, unknown 0010 -> APInt with unknown mapped to 0 = 1000 = 8 (signed: -8)
  // CHECK: hw.constant -8 : i4
  %cz = moore.constant b10Z0 : l4

  // Mixed X and Z - both map to 0
  // b10XZ = value 1001, unknown 0011 -> APInt with unknown mapped to 0 = 1000 = 8 (signed: -8)
  // CHECK: hw.constant -8 : i4
  %cxz = moore.constant b10XZ : l4

  // All X constant
  // bXXXX = value 0000, unknown 1111 -> APInt = 0
  // CHECK: hw.constant 0 : i4
  %allx = moore.constant bXXXX : l4

  // All Z constant
  // bZZZZ = value 1111, unknown 1111 -> APInt = 1111 & ~1111 = 0
  // CHECK: hw.constant 0 : i4
  %allz = moore.constant bZZZZ : l4

  return
}

// CHECK-LABEL: func @TwoStateConstants
func.func @TwoStateConstants() {
  // Two-valued constants should work as before
  // CHECK: hw.constant 5 : i4
  %c5 = moore.constant 5 : !moore.i4

  // CHECK: hw.constant -1 : i8
  %cff = moore.constant 255 : !moore.i8

  return
}

// Test logic operations with 4-state types (types are lowered to iN)

// CHECK-LABEL: func @FourStateLogicOps
// CHECK-SAME: (%arg0: i4, %arg1: i4)
func.func @FourStateLogicOps(%a: !moore.l4, %b: !moore.l4) {
  // AND operation
  // CHECK: comb.and %arg0, %arg1 : i4
  %and = moore.and %a, %b : l4

  // OR operation
  // CHECK: comb.or %arg0, %arg1 : i4
  %or = moore.or %a, %b : l4

  // XOR operation
  // CHECK: comb.xor %arg0, %arg1 : i4
  %xor = moore.xor %a, %b : l4

  // NOT operation
  // CHECK: %[[ONES:.*]] = hw.constant -1 : i4
  // CHECK: comb.xor %arg0, %[[ONES]] : i4
  %not = moore.not %a : l4

  return
}

// Test that two-valued types still work correctly

// CHECK-LABEL: func @TwoStateLogicOps
// CHECK-SAME: (%arg0: i4, %arg1: i4)
func.func @TwoStateLogicOps(%a: !moore.i4, %b: !moore.i4) {
  // AND operation
  // CHECK: comb.and %arg0, %arg1 : i4
  %and = moore.and %a, %b : i4

  // OR operation
  // CHECK: comb.or %arg0, %arg1 : i4
  %or = moore.or %a, %b : i4

  // XOR operation
  // CHECK: comb.xor %arg0, %arg1 : i4
  %xor = moore.xor %a, %b : i4

  // NOT operation
  // CHECK: %[[ONES:.*]] = hw.constant -1 : i4
  // CHECK: comb.xor %arg0, %[[ONES]] : i4
  %not = moore.not %a : i4

  return
}

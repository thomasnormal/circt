// RUN: circt-opt --canonicalize %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Conversion Folding Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_from_to_fourstate
func.func @fold_from_to_fourstate(%arg0: i8) -> i8 {
  %logic = arc.to_fourstate %arg0 : (i8) -> !arc.logic<8>
  // Folding from_fourstate(to_fourstate(x)) -> x
  // CHECK-NOT: arc.to_fourstate
  // CHECK-NOT: arc.from_fourstate
  // CHECK: return %arg0 : i8
  %result = arc.from_fourstate %logic : (!arc.logic<8>) -> i8
  return %result : i8
}

//===----------------------------------------------------------------------===//
// IsX Folding Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_is_x_constant
func.func @fold_is_x_constant() -> i1 {
  %c42 = arc.fourstate.constant 42 : !arc.logic<8>
  // Constants have no X/Z, so is_x should fold to false
  // CHECK: %[[FALSE:.*]] = arith.constant false
  // CHECK: return %[[FALSE]]
  %result = arc.is_x %c42 : !arc.logic<8>
  return %result : i1
}

// CHECK-LABEL: func.func @fold_is_x_from_to
func.func @fold_is_x_from_to(%arg0: i8) -> i1 {
  %logic = arc.to_fourstate %arg0 : (i8) -> !arc.logic<8>
  // Values from to_fourstate have no X/Z
  // CHECK: %[[FALSE:.*]] = arith.constant false
  // CHECK: return %[[FALSE]]
  %result = arc.is_x %logic : !arc.logic<8>
  return %result : i1
}

// CHECK-LABEL: func.func @fold_is_x_x_value
func.func @fold_is_x_x_value() -> i1 {
  %x = arc.fourstate.x : !arc.logic<8>
  // X values always have X
  // CHECK: %[[TRUE:.*]] = arith.constant true
  // CHECK: return %[[TRUE]]
  %result = arc.is_x %x : !arc.logic<8>
  return %result : i1
}

// CHECK-LABEL: func.func @fold_is_x_z_value
func.func @fold_is_x_z_value() -> i1 {
  %z = arc.fourstate.z : !arc.logic<8>
  // Z values have X/Z bits
  // CHECK: %[[TRUE:.*]] = arith.constant true
  // CHECK: return %[[TRUE]]
  %result = arc.is_x %z : !arc.logic<8>
  return %result : i1
}

//===----------------------------------------------------------------------===//
// Bitwise Logic Folding Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_and_same
func.func @fold_and_same(%arg0: !arc.logic<8>) -> !arc.logic<8> {
  // a & a = a
  // CHECK-NOT: arc.fourstate.and
  // CHECK: return %arg0
  %result = arc.fourstate.and %arg0, %arg0 : !arc.logic<8>
  return %result : !arc.logic<8>
}

// CHECK-LABEL: func.func @fold_and_zero
func.func @fold_and_zero(%arg0: !arc.logic<8>) -> !arc.logic<8> {
  %c0 = arc.fourstate.constant 0 : !arc.logic<8>
  // 0 & a = 0
  // CHECK: %[[ZERO:.*]] = arc.fourstate.constant 0
  // CHECK: return %[[ZERO]]
  %result = arc.fourstate.and %c0, %arg0 : !arc.logic<8>
  return %result : !arc.logic<8>
}

// CHECK-LABEL: func.func @fold_or_same
func.func @fold_or_same(%arg0: !arc.logic<8>) -> !arc.logic<8> {
  // a | a = a
  // CHECK-NOT: arc.fourstate.or
  // CHECK: return %arg0
  %result = arc.fourstate.or %arg0, %arg0 : !arc.logic<8>
  return %result : !arc.logic<8>
}

// CHECK-LABEL: func.func @fold_or_all_ones
func.func @fold_or_all_ones(%arg0: !arc.logic<8>) -> !arc.logic<8> {
  %c255 = arc.fourstate.constant 255 : !arc.logic<8>
  // all_ones | a = all_ones
  // CHECK: %[[ONES:.*]] = arc.fourstate.constant 255
  // CHECK: return %[[ONES]]
  %result = arc.fourstate.or %c255, %arg0 : !arc.logic<8>
  return %result : !arc.logic<8>
}

// CHECK-LABEL: func.func @fold_xor_zero
func.func @fold_xor_zero(%arg0: !arc.logic<8>) -> !arc.logic<8> {
  %c0 = arc.fourstate.constant 0 : !arc.logic<8>
  // a ^ 0 = a
  // CHECK-NOT: arc.fourstate.xor
  // CHECK: return %arg0
  %result = arc.fourstate.xor %arg0, %c0 : !arc.logic<8>
  return %result : !arc.logic<8>
}

// CHECK-LABEL: func.func @fold_not_not
func.func @fold_not_not(%arg0: !arc.logic<8>) -> !arc.logic<8> {
  // ~~a = a
  %not1 = arc.fourstate.not %arg0 : !arc.logic<8>
  // CHECK-NOT: arc.fourstate.not
  // CHECK: return %arg0
  %result = arc.fourstate.not %not1 : !arc.logic<8>
  return %result : !arc.logic<8>
}

//===----------------------------------------------------------------------===//
// Arithmetic Folding Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_add_zero
func.func @fold_add_zero(%arg0: !arc.logic<8>) -> !arc.logic<8> {
  %c0 = arc.fourstate.constant 0 : !arc.logic<8>
  // a + 0 = a
  // CHECK-NOT: arc.fourstate.add
  // CHECK: return %arg0
  %result = arc.fourstate.add %arg0, %c0 : !arc.logic<8>
  return %result : !arc.logic<8>
}

// CHECK-LABEL: func.func @fold_sub_zero
func.func @fold_sub_zero(%arg0: !arc.logic<8>) -> !arc.logic<8> {
  %c0 = arc.fourstate.constant 0 : !arc.logic<8>
  // a - 0 = a
  // CHECK-NOT: arc.fourstate.sub
  // CHECK: return %arg0
  %result = arc.fourstate.sub %arg0, %c0 : !arc.logic<8>
  return %result : !arc.logic<8>
}

// CHECK-LABEL: func.func @fold_mul_zero
func.func @fold_mul_zero(%arg0: !arc.logic<8>) -> !arc.logic<8> {
  %c0 = arc.fourstate.constant 0 : !arc.logic<8>
  // a * 0 = 0
  // CHECK: %[[ZERO:.*]] = arc.fourstate.constant 0
  // CHECK: return %[[ZERO]]
  %result = arc.fourstate.mul %arg0, %c0 : !arc.logic<8>
  return %result : !arc.logic<8>
}

// CHECK-LABEL: func.func @fold_mul_one
func.func @fold_mul_one(%arg0: !arc.logic<8>) -> !arc.logic<8> {
  %c1 = arc.fourstate.constant 1 : !arc.logic<8>
  // a * 1 = a
  // CHECK-NOT: arc.fourstate.mul
  // CHECK: return %arg0
  %result = arc.fourstate.mul %arg0, %c1 : !arc.logic<8>
  return %result : !arc.logic<8>
}

//===----------------------------------------------------------------------===//
// Comparison Folding Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_case_eq_same
func.func @fold_case_eq_same(%arg0: !arc.logic<8>) -> i1 {
  // a === a is always true (even for X/Z)
  // CHECK: %[[TRUE:.*]] = arith.constant true
  // CHECK: return %[[TRUE]]
  %result = arc.fourstate.case_eq %arg0, %arg0 : !arc.logic<8>
  return %result : i1
}

// CHECK-LABEL: func.func @fold_case_ne_same
func.func @fold_case_ne_same(%arg0: !arc.logic<8>) -> i1 {
  // a !== a is always false (even for X/Z)
  // CHECK: %[[FALSE:.*]] = arith.constant false
  // CHECK: return %[[FALSE]]
  %result = arc.fourstate.case_ne %arg0, %arg0 : !arc.logic<8>
  return %result : i1
}

//===----------------------------------------------------------------------===//
// Bit Manipulation Folding Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_concat_single
func.func @fold_concat_single(%arg0: !arc.logic<8>) -> !arc.logic<8> {
  // Concat of single value is identity
  // CHECK-NOT: arc.fourstate.concat
  // CHECK: return %arg0
  %result = arc.fourstate.concat %arg0 : !arc.logic<8> -> !arc.logic<8>
  return %result : !arc.logic<8>
}

// CHECK-LABEL: func.func @fold_extract_full
func.func @fold_extract_full(%arg0: !arc.logic<8>) -> !arc.logic<8> {
  // Extract of full width from 0 is identity
  // CHECK-NOT: arc.fourstate.extract
  // CHECK: return %arg0
  %result = arc.fourstate.extract %arg0 from 0 : !arc.logic<8> -> !arc.logic<8>
  return %result : !arc.logic<8>
}

// CHECK-LABEL: func.func @fold_replicate_same_width
func.func @fold_replicate_same_width(%arg0: !arc.logic<8>) -> !arc.logic<8> {
  // Replicate 1x is identity
  // CHECK-NOT: arc.fourstate.replicate
  // CHECK: return %arg0
  %result = arc.fourstate.replicate %arg0 : (!arc.logic<8>) -> !arc.logic<8>
  return %result : !arc.logic<8>
}

//===----------------------------------------------------------------------===//
// Mux Folding Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_mux_same_values
func.func @fold_mux_same_values(%cond: !arc.logic<1>, %arg0: !arc.logic<8>) -> !arc.logic<8> {
  // mux(cond, a, a) = a
  // CHECK-NOT: arc.fourstate.mux
  // CHECK: return %arg0
  %result = arc.fourstate.mux %cond, %arg0, %arg0 : !arc.logic<8>
  return %result : !arc.logic<8>
}

// CHECK-LABEL: func.func @fold_mux_true_condition
func.func @fold_mux_true_condition(%true_val: !arc.logic<8>, %false_val: !arc.logic<8>) -> !arc.logic<8> {
  %c1 = arc.fourstate.constant 1 : !arc.logic<1>
  // mux(1, a, b) = a
  // CHECK-NOT: arc.fourstate.mux
  // CHECK: return %arg0
  %result = arc.fourstate.mux %c1, %true_val, %false_val : !arc.logic<8>
  return %result : !arc.logic<8>
}

// CHECK-LABEL: func.func @fold_mux_false_condition
func.func @fold_mux_false_condition(%true_val: !arc.logic<8>, %false_val: !arc.logic<8>) -> !arc.logic<8> {
  %c0 = arc.fourstate.constant 0 : !arc.logic<1>
  // mux(0, a, b) = b
  // CHECK-NOT: arc.fourstate.mux
  // CHECK: return %arg1
  %result = arc.fourstate.mux %c0, %true_val, %false_val : !arc.logic<8>
  return %result : !arc.logic<8>
}

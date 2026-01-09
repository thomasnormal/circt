// RUN: circt-opt --verify-diagnostics --verify-roundtrip %s | circt-opt | FileCheck %s

//===----------------------------------------------------------------------===//
// FourState Type Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fourstate_types
func.func @fourstate_types(%arg0: !arc.logic<1>, %arg1: !arc.logic<8>, %arg2: !arc.logic<32>) {
  // CHECK: %arg0: !arc.logic<1>
  // CHECK: %arg1: !arc.logic<8>
  // CHECK: %arg2: !arc.logic<32>
  return
}

//===----------------------------------------------------------------------===//
// 4-State Constant Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fourstate_constants
func.func @fourstate_constants() -> (!arc.logic<8>, !arc.logic<8>, !arc.logic<8>) {
  // CHECK: arc.fourstate.constant 42 : !arc.logic<8>
  %c42 = arc.fourstate.constant 42 : !arc.logic<8>

  // CHECK: arc.fourstate.x : !arc.logic<8>
  %x = arc.fourstate.x : !arc.logic<8>

  // CHECK: arc.fourstate.z : !arc.logic<8>
  %z = arc.fourstate.z : !arc.logic<8>

  return %c42, %x, %z : !arc.logic<8>, !arc.logic<8>, !arc.logic<8>
}

// CHECK-LABEL: func.func @fourstate_constant_widths
func.func @fourstate_constant_widths() -> (!arc.logic<1>, !arc.logic<16>, !arc.logic<64>) {
  // CHECK: arc.fourstate.constant 1 : !arc.logic<1>
  %c1 = arc.fourstate.constant 1 : !arc.logic<1>

  // CHECK: arc.fourstate.constant 1234 : !arc.logic<16>
  %c1234 = arc.fourstate.constant 1234 : !arc.logic<16>

  // CHECK: arc.fourstate.constant 123456789 : !arc.logic<64>
  %cbig = arc.fourstate.constant 123456789 : !arc.logic<64>

  return %c1, %c1234, %cbig : !arc.logic<1>, !arc.logic<16>, !arc.logic<64>
}

//===----------------------------------------------------------------------===//
// Type Conversion Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fourstate_conversions
func.func @fourstate_conversions(%int8: i8, %int32: i32) -> (i8, i32) {
  // CHECK: arc.to_fourstate %arg0 : (i8) -> !arc.logic<8>
  %logic8 = arc.to_fourstate %int8 : (i8) -> !arc.logic<8>

  // CHECK: arc.to_fourstate %arg1 : (i32) -> !arc.logic<32>
  %logic32 = arc.to_fourstate %int32 : (i32) -> !arc.logic<32>

  // CHECK: arc.from_fourstate {{%.*}} : (!arc.logic<8>) -> i8
  %result8 = arc.from_fourstate %logic8 : (!arc.logic<8>) -> i8

  // CHECK: arc.from_fourstate {{%.*}} : (!arc.logic<32>) -> i32
  %result32 = arc.from_fourstate %logic32 : (!arc.logic<32>) -> i32

  return %result8, %result32 : i8, i32
}

// CHECK-LABEL: func.func @fourstate_is_x
func.func @fourstate_is_x(%logic: !arc.logic<8>) -> (i1, i1, i1) {
  // CHECK: arc.is_x {{%.*}} : !arc.logic<8>
  %has_x = arc.is_x %logic : !arc.logic<8>

  // Test with known non-X value
  %c42 = arc.fourstate.constant 42 : !arc.logic<8>
  // CHECK: arc.is_x {{%.*}} : !arc.logic<8>
  %no_x = arc.is_x %c42 : !arc.logic<8>

  // Test with X value
  %x = arc.fourstate.x : !arc.logic<8>
  // CHECK: arc.is_x {{%.*}} : !arc.logic<8>
  %all_x = arc.is_x %x : !arc.logic<8>

  return %has_x, %no_x, %all_x : i1, i1, i1
}

//===----------------------------------------------------------------------===//
// Bitwise Logic Operation Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fourstate_bitwise_ops
func.func @fourstate_bitwise_ops(%a: !arc.logic<8>, %b: !arc.logic<8>) -> (!arc.logic<8>, !arc.logic<8>, !arc.logic<8>, !arc.logic<8>) {
  // CHECK: arc.fourstate.and {{%.*}}, {{%.*}} : !arc.logic<8>
  %and_result = arc.fourstate.and %a, %b : !arc.logic<8>

  // CHECK: arc.fourstate.or {{%.*}}, {{%.*}} : !arc.logic<8>
  %or_result = arc.fourstate.or %a, %b : !arc.logic<8>

  // CHECK: arc.fourstate.xor {{%.*}}, {{%.*}} : !arc.logic<8>
  %xor_result = arc.fourstate.xor %a, %b : !arc.logic<8>

  // CHECK: arc.fourstate.not {{%.*}} : !arc.logic<8>
  %not_result = arc.fourstate.not %a : !arc.logic<8>

  return %and_result, %or_result, %xor_result, %not_result : !arc.logic<8>, !arc.logic<8>, !arc.logic<8>, !arc.logic<8>
}

// CHECK-LABEL: func.func @fourstate_bitwise_chain
func.func @fourstate_bitwise_chain(%a: !arc.logic<8>, %b: !arc.logic<8>, %c: !arc.logic<8>) -> !arc.logic<8> {
  // Test chaining of operations
  // CHECK: arc.fourstate.and
  %t1 = arc.fourstate.and %a, %b : !arc.logic<8>
  // CHECK: arc.fourstate.or
  %t2 = arc.fourstate.or %t1, %c : !arc.logic<8>
  // CHECK: arc.fourstate.not
  %result = arc.fourstate.not %t2 : !arc.logic<8>
  return %result : !arc.logic<8>
}

//===----------------------------------------------------------------------===//
// Arithmetic Operation Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fourstate_arithmetic_ops
func.func @fourstate_arithmetic_ops(%a: !arc.logic<8>, %b: !arc.logic<8>) -> (!arc.logic<8>, !arc.logic<8>, !arc.logic<8>) {
  // CHECK: arc.fourstate.add {{%.*}}, {{%.*}} : !arc.logic<8>
  %add_result = arc.fourstate.add %a, %b : !arc.logic<8>

  // CHECK: arc.fourstate.sub {{%.*}}, {{%.*}} : !arc.logic<8>
  %sub_result = arc.fourstate.sub %a, %b : !arc.logic<8>

  // CHECK: arc.fourstate.mul {{%.*}}, {{%.*}} : !arc.logic<8>
  %mul_result = arc.fourstate.mul %a, %b : !arc.logic<8>

  return %add_result, %sub_result, %mul_result : !arc.logic<8>, !arc.logic<8>, !arc.logic<8>
}

//===----------------------------------------------------------------------===//
// Comparison Operation Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fourstate_comparison_ops
func.func @fourstate_comparison_ops(%a: !arc.logic<8>, %b: !arc.logic<8>) -> (!arc.logic<1>, !arc.logic<1>, i1, i1) {
  // CHECK: arc.fourstate.eq {{%.*}}, {{%.*}} : !arc.logic<8> -> !arc.logic<1>
  %eq_result = arc.fourstate.eq %a, %b : !arc.logic<8> -> !arc.logic<1>

  // CHECK: arc.fourstate.ne {{%.*}}, {{%.*}} : !arc.logic<8> -> !arc.logic<1>
  %ne_result = arc.fourstate.ne %a, %b : !arc.logic<8> -> !arc.logic<1>

  // CHECK: arc.fourstate.case_eq {{%.*}}, {{%.*}} : !arc.logic<8>
  %case_eq_result = arc.fourstate.case_eq %a, %b : !arc.logic<8>

  // CHECK: arc.fourstate.case_ne {{%.*}}, {{%.*}} : !arc.logic<8>
  %case_ne_result = arc.fourstate.case_ne %a, %b : !arc.logic<8>

  return %eq_result, %ne_result, %case_eq_result, %case_ne_result : !arc.logic<1>, !arc.logic<1>, i1, i1
}

//===----------------------------------------------------------------------===//
// Bit Manipulation Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fourstate_concat
func.func @fourstate_concat(%a: !arc.logic<4>, %b: !arc.logic<4>) -> !arc.logic<8> {
  // CHECK: arc.fourstate.concat {{%.*}}, {{%.*}} : !arc.logic<4>, !arc.logic<4> -> !arc.logic<8>
  %result = arc.fourstate.concat %a, %b : !arc.logic<4>, !arc.logic<4> -> !arc.logic<8>
  return %result : !arc.logic<8>
}

// CHECK-LABEL: func.func @fourstate_concat_multi
func.func @fourstate_concat_multi(%a: !arc.logic<2>, %b: !arc.logic<2>, %c: !arc.logic<4>) -> !arc.logic<8> {
  // CHECK: arc.fourstate.concat {{%.*}}, {{%.*}}, {{%.*}} : !arc.logic<2>, !arc.logic<2>, !arc.logic<4> -> !arc.logic<8>
  %result = arc.fourstate.concat %a, %b, %c : !arc.logic<2>, !arc.logic<2>, !arc.logic<4> -> !arc.logic<8>
  return %result : !arc.logic<8>
}

// CHECK-LABEL: func.func @fourstate_extract
func.func @fourstate_extract(%input: !arc.logic<16>) -> (!arc.logic<4>, !arc.logic<8>) {
  // CHECK: arc.fourstate.extract {{%.*}} from 0 : !arc.logic<16> -> !arc.logic<4>
  %low = arc.fourstate.extract %input from 0 : !arc.logic<16> -> !arc.logic<4>

  // CHECK: arc.fourstate.extract {{%.*}} from 4 : !arc.logic<16> -> !arc.logic<8>
  %mid = arc.fourstate.extract %input from 4 : !arc.logic<16> -> !arc.logic<8>

  return %low, %mid : !arc.logic<4>, !arc.logic<8>
}

// CHECK-LABEL: func.func @fourstate_replicate
func.func @fourstate_replicate(%input: !arc.logic<4>) -> (!arc.logic<8>, !arc.logic<16>) {
  // CHECK: arc.fourstate.replicate {{%.*}} : (!arc.logic<4>) -> !arc.logic<8>
  %rep2 = arc.fourstate.replicate %input : (!arc.logic<4>) -> !arc.logic<8>

  // CHECK: arc.fourstate.replicate {{%.*}} : (!arc.logic<4>) -> !arc.logic<16>
  %rep4 = arc.fourstate.replicate %input : (!arc.logic<4>) -> !arc.logic<16>

  return %rep2, %rep4 : !arc.logic<8>, !arc.logic<16>
}

//===----------------------------------------------------------------------===//
// Conditional Operation Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fourstate_mux
func.func @fourstate_mux(%cond: !arc.logic<1>, %true_val: !arc.logic<8>, %false_val: !arc.logic<8>) -> !arc.logic<8> {
  // CHECK: arc.fourstate.mux {{%.*}}, {{%.*}}, {{%.*}} : !arc.logic<1>, !arc.logic<8>
  %result = arc.fourstate.mux %cond, %true_val, %false_val : !arc.logic<1>, !arc.logic<8>
  return %result : !arc.logic<8>
}

// CHECK-LABEL: func.func @fourstate_mux_chain
func.func @fourstate_mux_chain(%c1: !arc.logic<1>, %c2: !arc.logic<1>, %a: !arc.logic<8>, %b: !arc.logic<8>, %c: !arc.logic<8>) -> !arc.logic<8> {
  // Chained mux (like a priority encoder)
  // CHECK: arc.fourstate.mux
  %t1 = arc.fourstate.mux %c1, %a, %b : !arc.logic<1>, !arc.logic<8>
  // CHECK: arc.fourstate.mux
  %result = arc.fourstate.mux %c2, %t1, %c : !arc.logic<1>, !arc.logic<8>
  return %result : !arc.logic<8>
}

//===----------------------------------------------------------------------===//
// Integration Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fourstate_full_adder
func.func @fourstate_full_adder(%a: !arc.logic<1>, %b: !arc.logic<1>, %cin: !arc.logic<1>) -> (!arc.logic<1>, !arc.logic<1>) {
  // Full adder implementation using 4-state logic
  // sum = a ^ b ^ cin
  // CHECK: arc.fourstate.xor
  %t1 = arc.fourstate.xor %a, %b : !arc.logic<1>
  // CHECK: arc.fourstate.xor
  %sum = arc.fourstate.xor %t1, %cin : !arc.logic<1>

  // cout = (a & b) | (cin & (a ^ b))
  // CHECK: arc.fourstate.and
  %t2 = arc.fourstate.and %a, %b : !arc.logic<1>
  // CHECK: arc.fourstate.and
  %t3 = arc.fourstate.and %cin, %t1 : !arc.logic<1>
  // CHECK: arc.fourstate.or
  %cout = arc.fourstate.or %t2, %t3 : !arc.logic<1>

  return %sum, %cout : !arc.logic<1>, !arc.logic<1>
}

// CHECK-LABEL: func.func @fourstate_with_arc_define
arc.define @logic_and_arc(%a: !arc.logic<8>, %b: !arc.logic<8>) -> !arc.logic<8> {
  %result = arc.fourstate.and %a, %b : !arc.logic<8>
  arc.output %result : !arc.logic<8>
}

// CHECK-LABEL: func.func @fourstate_arc_call
func.func @fourstate_arc_call(%a: !arc.logic<8>, %b: !arc.logic<8>) -> !arc.logic<8> {
  // CHECK: arc.call @logic_and_arc
  %result = arc.call @logic_and_arc(%a, %b) : (!arc.logic<8>, !arc.logic<8>) -> !arc.logic<8>
  return %result : !arc.logic<8>
}

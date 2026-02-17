// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// Test that bool_cast of f64 (real) types works correctly.
// This tests the fix for the crash when processing covergroup methods
// that return f64 (like get_coverage()) and are then used in boolean
// contexts.

// Covergroup declaration needed for get_coverage
moore.covergroup.decl @TestCG {
  moore.coverpoint.decl @data : !moore.i8 {}
}

// CHECK-LABEL: func @test_boolcast_f64
// CHECK-SAME: -> i1
func.func @test_boolcast_f64() -> !moore.i1 {
  %cg = moore.covergroup.inst @TestCG : !moore.covergroup<@TestCG>
  // CHECK: [[COV:%.+]] = llvm.call @__moore_covergroup_get_coverage
  %cov = moore.covergroup.get_coverage %cg : !moore.covergroup<@TestCG> -> !moore.f64
  // CHECK: [[ZERO:%.+]] = arith.constant 0.000000e+00 : f64
  // CHECK: [[RESULT:%.+]] = arith.cmpf une, [[COV]], [[ZERO]] : f64
  // CHECK: return [[RESULT]] : i1
  %result = moore.bool_cast %cov : !moore.f64 -> !moore.i1
  return %result : !moore.i1
}

// CHECK-LABEL: func @test_boolcast_f64_direct
// CHECK-SAME: (%[[ARG:.*]]: f64) -> i1
func.func @test_boolcast_f64_direct(%real: !moore.f64) -> !moore.i1 {
  // CHECK: [[ZERO:%.+]] = arith.constant 0.000000e+00 : f64
  // CHECK: [[RESULT:%.+]] = arith.cmpf une, %[[ARG]], [[ZERO]] : f64
  // CHECK: return [[RESULT]] : i1
  %result = moore.bool_cast %real : !moore.f64 -> !moore.i1
  return %result : !moore.i1
}

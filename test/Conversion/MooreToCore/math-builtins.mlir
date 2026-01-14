// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// Trigonometric Functions (IEEE 1800-2017 Section 20.8.2)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_sin
func.func @test_sin(%arg0: !moore.f64) -> !moore.f64 {
  // CHECK: math.sin %arg0 : f64
  %0 = moore.builtin.sin %arg0 : !moore.f64
  return %0 : !moore.f64
}

// CHECK-LABEL: func.func @test_cos
func.func @test_cos(%arg0: !moore.f64) -> !moore.f64 {
  // CHECK: math.cos %arg0 : f64
  %0 = moore.builtin.cos %arg0 : !moore.f64
  return %0 : !moore.f64
}

// CHECK-LABEL: func.func @test_tan
func.func @test_tan(%arg0: !moore.f64) -> !moore.f64 {
  // CHECK: math.tan %arg0 : f64
  %0 = moore.builtin.tan %arg0 : !moore.f64
  return %0 : !moore.f64
}

// CHECK-LABEL: func.func @test_asin
func.func @test_asin(%arg0: !moore.f64) -> !moore.f64 {
  // CHECK: math.asin %arg0 : f64
  %0 = moore.builtin.asin %arg0 : !moore.f64
  return %0 : !moore.f64
}

// CHECK-LABEL: func.func @test_acos
func.func @test_acos(%arg0: !moore.f64) -> !moore.f64 {
  // CHECK: math.acos %arg0 : f64
  %0 = moore.builtin.acos %arg0 : !moore.f64
  return %0 : !moore.f64
}

// CHECK-LABEL: func.func @test_atan
func.func @test_atan(%arg0: !moore.f64) -> !moore.f64 {
  // CHECK: math.atan %arg0 : f64
  %0 = moore.builtin.atan %arg0 : !moore.f64
  return %0 : !moore.f64
}

//===----------------------------------------------------------------------===//
// Hyperbolic Functions (IEEE 1800-2017 Section 20.8.2)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_sinh
func.func @test_sinh(%arg0: !moore.f64) -> !moore.f64 {
  // CHECK: math.sinh %arg0 : f64
  %0 = moore.builtin.sinh %arg0 : !moore.f64
  return %0 : !moore.f64
}

// CHECK-LABEL: func.func @test_cosh
func.func @test_cosh(%arg0: !moore.f64) -> !moore.f64 {
  // CHECK: math.cosh %arg0 : f64
  %0 = moore.builtin.cosh %arg0 : !moore.f64
  return %0 : !moore.f64
}

// CHECK-LABEL: func.func @test_tanh
func.func @test_tanh(%arg0: !moore.f64) -> !moore.f64 {
  // CHECK: math.tanh %arg0 : f64
  %0 = moore.builtin.tanh %arg0 : !moore.f64
  return %0 : !moore.f64
}

// CHECK-LABEL: func.func @test_asinh
func.func @test_asinh(%arg0: !moore.f64) -> !moore.f64 {
  // CHECK: math.asinh %arg0 : f64
  %0 = moore.builtin.asinh %arg0 : !moore.f64
  return %0 : !moore.f64
}

// CHECK-LABEL: func.func @test_acosh
func.func @test_acosh(%arg0: !moore.f64) -> !moore.f64 {
  // CHECK: math.acosh %arg0 : f64
  %0 = moore.builtin.acosh %arg0 : !moore.f64
  return %0 : !moore.f64
}

// CHECK-LABEL: func.func @test_atanh
func.func @test_atanh(%arg0: !moore.f64) -> !moore.f64 {
  // CHECK: math.atanh %arg0 : f64
  %0 = moore.builtin.atanh %arg0 : !moore.f64
  return %0 : !moore.f64
}

//===----------------------------------------------------------------------===//
// Exponential and Logarithmic Functions (IEEE 1800-2017 Section 20.8.2)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_exp
func.func @test_exp(%arg0: !moore.f64) -> !moore.f64 {
  // CHECK: math.exp %arg0 : f64
  %0 = moore.builtin.exp %arg0 : !moore.f64
  return %0 : !moore.f64
}

// CHECK-LABEL: func.func @test_ln
func.func @test_ln(%arg0: !moore.f64) -> !moore.f64 {
  // CHECK: math.log %arg0 : f64
  %0 = moore.builtin.ln %arg0 : !moore.f64
  return %0 : !moore.f64
}

// CHECK-LABEL: func.func @test_log10
func.func @test_log10(%arg0: !moore.f64) -> !moore.f64 {
  // CHECK: math.log10 %arg0 : f64
  %0 = moore.builtin.log10 %arg0 : !moore.f64
  return %0 : !moore.f64
}

// CHECK-LABEL: func.func @test_sqrt
func.func @test_sqrt(%arg0: !moore.f64) -> !moore.f64 {
  // CHECK: math.sqrt %arg0 : f64
  %0 = moore.builtin.sqrt %arg0 : !moore.f64
  return %0 : !moore.f64
}

//===----------------------------------------------------------------------===//
// Rounding Functions (IEEE 1800-2017 Section 20.8.2)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_floor
func.func @test_floor(%arg0: !moore.f64) -> !moore.f64 {
  // CHECK: math.floor %arg0 : f64
  %0 = moore.builtin.floor %arg0 : !moore.f64
  return %0 : !moore.f64
}

// CHECK-LABEL: func.func @test_ceil
func.func @test_ceil(%arg0: !moore.f64) -> !moore.f64 {
  // CHECK: math.ceil %arg0 : f64
  %0 = moore.builtin.ceil %arg0 : !moore.f64
  return %0 : !moore.f64
}

//===----------------------------------------------------------------------===//
// Integer Math Functions (IEEE 1800-2017 Section 20.8.1)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_clog2
func.func @test_clog2(%arg0: !moore.i32) -> !moore.i32 {
  // CHECK: %[[C0:.*]] = hw.constant 0 : i32
  // CHECK: %[[C1:.*]] = hw.constant 1 : i32
  // CHECK: %[[CWIDTH:.*]] = hw.constant 32 : i32
  // CHECK: %[[SUB:.*]] = comb.sub %arg0, %[[C1]] : i32
  // CHECK: %[[CTLZ:.*]] = "llvm.intr.ctlz"(%[[SUB]]) <{is_zero_poison = false}> : (i32) -> i32
  // CHECK: %[[RESULT:.*]] = comb.sub %[[CWIDTH]], %[[CTLZ]] : i32
  // CHECK: %[[CMP:.*]] = comb.icmp ule %arg0, %[[C1]] : i32
  // CHECK: comb.mux %[[CMP]], %[[C0]], %[[RESULT]] : i32
  %0 = moore.builtin.clog2 %arg0 : !moore.i32
  return %0 : !moore.i32
}

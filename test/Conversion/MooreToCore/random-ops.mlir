// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_urandom() -> i32
// CHECK-DAG: llvm.func @__moore_urandom_seeded(i32) -> i32
// CHECK-DAG: llvm.func @__moore_urandom_range(i32, i32) -> i32
// CHECK-DAG: llvm.func @__moore_random() -> i32
// CHECK-DAG: llvm.func @__moore_random_seeded(i32) -> i32
// CHECK-DAG: llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32

//===----------------------------------------------------------------------===//
// $urandom Operation
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_urandom
moore.module @test_urandom(out result: !moore.i32) {
  // CHECK: %[[RESULT:.*]] = llvm.call @__moore_urandom() : () -> i32
  %urandom = moore.builtin.urandom
  moore.output %urandom : !moore.i32
}

//===----------------------------------------------------------------------===//
// $urandom with seed Operation
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_urandom_seeded
moore.module @test_urandom_seeded(in %seed: !moore.i32, out result: !moore.i32) {
  // CHECK: %[[RESULT:.*]] = llvm.call @__moore_urandom_seeded(%{{.*}}) : (i32) -> i32
  %urandom = moore.builtin.urandom seed %seed
  moore.output %urandom : !moore.i32
}

//===----------------------------------------------------------------------===//
// $urandom_range Operation (with max only)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_urandom_range_max
moore.module @test_urandom_range_max(in %maxval: !moore.i32, out result: !moore.i32) {
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
  // CHECK: %[[RESULT:.*]] = llvm.call @__moore_urandom_range(%{{.*}}, %[[ZERO]]) : (i32, i32) -> i32
  %urandom_range = moore.builtin.urandom_range %maxval
  moore.output %urandom_range : !moore.i32
}

//===----------------------------------------------------------------------===//
// $urandom_range Operation (with max and min)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_urandom_range_max_min
moore.module @test_urandom_range_max_min(in %maxval: !moore.i32, in %minval: !moore.i32, out result: !moore.i32) {
  // CHECK: %[[RESULT:.*]] = llvm.call @__moore_urandom_range(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
  %urandom_range = moore.builtin.urandom_range %maxval, %minval
  moore.output %urandom_range : !moore.i32
}

//===----------------------------------------------------------------------===//
// $random Operation
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_random
moore.module @test_random(out result: !moore.i32) {
  // CHECK: %[[RESULT:.*]] = llvm.call @__moore_random() : () -> i32
  %random = moore.builtin.random
  moore.output %random : !moore.i32
}

//===----------------------------------------------------------------------===//
// $random with seed Operation
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_random_seeded
moore.module @test_random_seeded(in %seed: !moore.i32, out result: !moore.i32) {
  // CHECK: %[[RESULT:.*]] = llvm.call @__moore_random_seeded(%{{.*}}) : (i32) -> i32
  %random = moore.builtin.random seed %seed
  moore.output %random : !moore.i32
}

//===----------------------------------------------------------------------===//
// randomize() Operation
//===----------------------------------------------------------------------===//

// Class declaration for testing randomize
moore.class.classdecl @TestClass {
  moore.class.propertydecl @field1 : !moore.i32
  moore.class.propertydecl @field2 : !moore.l64
}

// CHECK-LABEL: func.func @test_randomize
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_randomize(%obj: !moore.class<@TestClass>) -> i1 {
  // The size is 24 bytes: 4 (i32) + 4 (padding) + 8 (l64 hi) + 8 (l64 lo) = 24
  // CHECK: llvm.call @__moore_randomize_basic(%[[OBJ]], {{.*}}) : (!llvm.ptr, i64) -> i32
  // CHECK: arith.trunci
  %success = moore.randomize %obj : !moore.class<@TestClass>
  return %success : i1
}

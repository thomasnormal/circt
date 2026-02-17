// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32
// CHECK-DAG: llvm.func @__moore_is_rand_enabled(!llvm.ptr, !llvm.ptr) -> i32

//===----------------------------------------------------------------------===//
// Distribution Constraint Support Tests
//===----------------------------------------------------------------------===//

/// Test class with simple distribution constraint using := (per-value) weights
/// Corresponds to SystemVerilog: constraint c { value dist { 0 := 10, 1 := 20, 2 := 30 }; }

moore.class.classdecl @SimpleDistClass {
  moore.class.propertydecl @value : !moore.i8 rand_mode rand
  moore.constraint.block @dist_c {
  ^bb0(%value: !moore.i8):
    // Distribution: 0 := 10, 1 := 20, 2 := 30
    // values: [0, 0, 1, 1, 2, 2] (as [low, high] pairs)
    // weights: [10, 20, 30]
    // perRange: [0, 0, 0] (all := per-value)
    moore.constraint.dist %value, [0, 0, 1, 1, 2, 2], [10, 20, 30], [0, 0, 0] : !moore.i8
  }
}

// CHECK-LABEL: func.func @test_simple_dist
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_simple_dist(%obj: !moore.class<@SimpleDistClass>) -> i1 {
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[OBJ]][0, 2]
  // CHECK: %[[SAVED:.*]] = llvm.load %[[GEP]] : !llvm.ptr -> i8
  // CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
  // CHECK: llvm.call @__moore_randomize_basic(%[[OBJ]], {{.*}}) : (!llvm.ptr, i64) -> i32
  // CHECK: arith.trunci {{.*}} : i32 to i1
  // CHECK: scf.if
  // CHECK:   llvm.store %[[SAVED]], %[[GEP]] : i8, !llvm.ptr
  // CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
  // CHECK: return %{{.*}} : i1
  %success = moore.randomize %obj : !moore.class<@SimpleDistClass>
  return %success : i1
}

/// Test class with range distribution using := (per-value) weight
/// Corresponds to SystemVerilog: constraint c { value dist { [0:10] := 50 }; }

moore.class.classdecl @RangeDistClass {
  moore.class.propertydecl @value : !moore.i8 rand_mode rand
  moore.constraint.block @dist_c {
  ^bb0(%value: !moore.i8):
    // Distribution: [0:10] := 50 (each value in [0,10] gets weight 50)
    moore.constraint.dist %value, [0, 10], [50], [0] : !moore.i8
  }
}

// CHECK-LABEL: func.func @test_range_dist
// CHECK: llvm.call @__moore_randomize_basic
// CHECK: return %{{.*}} : i1
func.func @test_range_dist(%obj: !moore.class<@RangeDistClass>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@RangeDistClass>
  return %success : i1
}

/// Test class with per-range distribution using :/ (divided) weight
/// Corresponds to SystemVerilog: constraint c { value dist { [1:5] :/ 100 }; }

moore.class.classdecl @PerRangeDistClass {
  moore.class.propertydecl @value : !moore.i8 rand_mode rand
  moore.constraint.block @dist_c {
  ^bb0(%value: !moore.i8):
    // Distribution: [1:5] :/ 100 (total weight 100 divided among values 1-5)
    // perRange: [1] (= :/ per-range)
    moore.constraint.dist %value, [1, 5], [100], [1] : !moore.i8
  }
}

// CHECK-LABEL: func.func @test_per_range_dist
// CHECK: llvm.call @__moore_randomize_basic
// CHECK: return %{{.*}} : i1
func.func @test_per_range_dist(%obj: !moore.class<@PerRangeDistClass>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@PerRangeDistClass>
  return %success : i1
}

/// Test class with mixed := and :/ weights
/// Corresponds to SystemVerilog: constraint c { value dist { 0 := 1, [1:10] :/ 4 }; }

moore.class.classdecl @MixedDistClass {
  moore.class.propertydecl @value : !moore.i8 rand_mode rand
  moore.constraint.block @dist_c {
  ^bb0(%value: !moore.i8):
    // Distribution: 0 := 1, [1:10] :/ 4
    // ranges: [0, 0], [1, 10]
    // weights: [1, 4]
    // perRange: [0, 1] (:= for first, :/ for second)
    moore.constraint.dist %value, [0, 0, 1, 10], [1, 4], [0, 1] : !moore.i8
  }
}

// CHECK-LABEL: func.func @test_mixed_dist
// CHECK: llvm.call @__moore_randomize_basic
// CHECK: return %{{.*}} : i1
func.func @test_mixed_dist(%obj: !moore.class<@MixedDistClass>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@MixedDistClass>
  return %success : i1
}

/// Test class with both dist constraint and inside constraint
/// Both fields are randomized via randomize_basic

moore.class.classdecl @DistAndInsideClass {
  moore.class.propertydecl @x : !moore.i8 rand_mode rand
  moore.class.propertydecl @y : !moore.i8 rand_mode rand
  moore.constraint.block @dist_c {
  ^bb0(%x: !moore.i8, %y: !moore.i8):
    // x uses dist constraint
    moore.constraint.dist %x, [1, 5, 10, 20], [3, 7], [0, 0] : !moore.i8
    // y uses inside constraint
    moore.constraint.inside %y, [100, 200] : !moore.i8
  }
}

// CHECK-LABEL: func.func @test_dist_and_inside
// CHECK: llvm.call @__moore_is_rand_enabled
// CHECK: llvm.call @__moore_is_rand_enabled
// CHECK: llvm.call @__moore_randomize_basic
// CHECK: return %{{.*}} : i1
func.func @test_dist_and_inside(%obj: !moore.class<@DistAndInsideClass>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@DistAndInsideClass>
  return %success : i1
}

/// Test that ConstraintDistOp is erased properly

moore.class.classdecl @DistErasedClass {
  moore.class.propertydecl @z : !moore.i32 rand_mode rand
  moore.constraint.block @erase_c {
  ^bb0(%z: !moore.i32):
    moore.constraint.dist %z, [0, 100], [1], [0] : !moore.i32
  }
}

// CHECK-LABEL: func.func @test_dist_erased
// CHECK-NOT: moore.constraint.dist
// CHECK-NOT: moore.constraint.block
// CHECK: llvm.call @__moore_randomize_basic
// CHECK: return %{{.*}} : i1
func.func @test_dist_erased(%obj: !moore.class<@DistErasedClass>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@DistErasedClass>
  return %success : i1
}

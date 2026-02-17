// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32
// CHECK-DAG: llvm.func @__moore_is_rand_enabled(!llvm.ptr, !llvm.ptr) -> i32

//===----------------------------------------------------------------------===//
// Range Constraint Support Tests
//===----------------------------------------------------------------------===//

/// Test class with simple range constraint using ConstraintInsideOp
/// Corresponds to SystemVerilog: constraint range_c { value inside {[1:99]}; }

moore.class.classdecl @RangeConstrainedClass {
  moore.class.propertydecl @value : !moore.i32 rand_mode rand
  moore.constraint.block @range_c {
  ^bb0(%value: !moore.i32):
    // Constraint: value inside {[1:99]} - single range [1, 99]
    moore.constraint.inside %value, [1, 99] : !moore.i32
  }
}

// CHECK-LABEL: func.func @test_range_constraint
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_range_constraint(%obj: !moore.class<@RangeConstrainedClass>) -> i1 {
  // Save old value, check rand_enabled, call randomize_basic, restore if disabled
  // CHECK: llvm.getelementptr %[[OBJ]][0, 2]
  // CHECK: llvm.load
  // CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}})
  // CHECK: llvm.call @__moore_randomize_basic(%[[OBJ]], {{.*}})
  // CHECK: arith.trunci {{.*}} : i32 to i1
  // CHECK: scf.if
  // CHECK:   llvm.store
  // CHECK: llvm.call @__moore_is_rand_enabled
  // CHECK: arith.ori
  // CHECK: arith.andi
  // CHECK: return {{.*}} : i1
  %success = moore.randomize %obj : !moore.class<@RangeConstrainedClass>
  return %success : i1
}

/// Test class without constraints - should use basic randomization only

moore.class.classdecl @UnconstrainedClass {
  moore.class.propertydecl @data : !moore.i32 rand_mode rand
}

// CHECK-LABEL: func.func @test_unconstrained
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_unconstrained(%obj: !moore.class<@UnconstrainedClass>) -> i1 {
  // CHECK: llvm.getelementptr %[[OBJ]][0, 2]
  // CHECK: llvm.load
  // CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}})
  // CHECK: llvm.call @__moore_randomize_basic(%[[OBJ]], {{.*}})
  // CHECK: arith.trunci {{.*}} : i32 to i1
  // CHECK: scf.if
  // CHECK:   llvm.store
  // CHECK: arith.ori
  // CHECK: arith.andi
  // CHECK: return
  %success = moore.randomize %obj : !moore.class<@UnconstrainedClass>
  return %success : i1
}

/// Test class with constraint block but empty body - should use basic randomization

moore.class.classdecl @EmptyConstraintClass {
  moore.class.propertydecl @x : !moore.i32 rand_mode rand
  moore.constraint.block @empty_constraint {
  }
}

// CHECK-LABEL: func.func @test_empty_constraint
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_empty_constraint(%obj: !moore.class<@EmptyConstraintClass>) -> i1 {
  // CHECK: llvm.call @__moore_randomize_basic(%[[OBJ]], {{.*}})
  // CHECK: arith.trunci {{.*}} : i32 to i1
  // CHECK: return
  %success = moore.randomize %obj : !moore.class<@EmptyConstraintClass>
  return %success : i1
}

/// Test class with multiple properties but only one constrained

moore.class.classdecl @PartialConstraintClass {
  moore.class.propertydecl @constrained : !moore.i32 rand_mode rand
  moore.class.propertydecl @unconstrained : !moore.i32 rand_mode rand
  moore.constraint.block @partial_c {
  ^bb0(%constrained: !moore.i32):
    // Only constrained property is in the constraint
    moore.constraint.inside %constrained, [10, 20] : !moore.i32
  }
}

// CHECK-LABEL: func.func @test_partial_constraint
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_partial_constraint(%obj: !moore.class<@PartialConstraintClass>) -> i1 {
  // Both properties are saved/restored based on rand_enabled
  // CHECK: llvm.getelementptr %[[OBJ]][0, 2]
  // CHECK: llvm.load
  // CHECK: llvm.call @__moore_is_rand_enabled
  // CHECK: llvm.getelementptr %[[OBJ]][0, 3]
  // CHECK: llvm.load
  // CHECK: llvm.call @__moore_is_rand_enabled
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK: scf.if
  // CHECK: scf.if
  // CHECK: arith.ori
  // CHECK: arith.ori
  // CHECK: arith.andi
  // CHECK: return
  %success = moore.randomize %obj : !moore.class<@PartialConstraintClass>
  return %success : i1
}

//===----------------------------------------------------------------------===//
// Constraint Expression Lowering Tests
//===----------------------------------------------------------------------===//

/// Test class with ConstraintExprOp - boolean constraint expressions
/// Corresponds to SystemVerilog: constraint c { x > 0; x < 100; }

moore.class.classdecl @ExprConstraintClass {
  moore.class.propertydecl @x : !moore.i32 rand_mode rand
  moore.constraint.block @expr_c {
  ^bb0(%x: !moore.i32):
    %c0 = moore.constant 0 : i32
    %c100 = moore.constant 100 : i32
    %gt = moore.sgt %x, %c0 : i32 -> i1
    moore.constraint.expr %gt : i1
    %lt = moore.slt %x, %c100 : i32 -> i1
    moore.constraint.expr %lt : i1
  }
}

// CHECK-LABEL: func.func @test_expr_constraint
// CHECK-NOT: moore.constraint.block
// CHECK-NOT: moore.constraint.expr
func.func @test_expr_constraint(%obj: !moore.class<@ExprConstraintClass>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@ExprConstraintClass>
  return %success : i1
}

/// Test class with ConstraintImplicationOp
/// Corresponds to SystemVerilog: constraint c { mode == 1 -> speed > 10; }

moore.class.classdecl @ImplicationConstraintClass {
  moore.class.propertydecl @mode : !moore.i32 rand_mode rand
  moore.class.propertydecl @speed : !moore.i32 rand_mode rand
  moore.constraint.block @impl_c {
  ^bb0(%mode: !moore.i32, %speed: !moore.i32):
    %c1 = moore.constant 1 : i32
    %c10 = moore.constant 10 : i32
    %mode_eq_1 = moore.eq %mode, %c1 : i32 -> i1
    moore.constraint.implication %mode_eq_1 : i1 {
      %speed_gt_10 = moore.sgt %speed, %c10 : i32 -> i1
      moore.constraint.expr %speed_gt_10 : i1
    }
  }
}

// CHECK-LABEL: func.func @test_implication_constraint
// CHECK-NOT: moore.constraint.block
// CHECK-NOT: moore.constraint.implication
// CHECK-NOT: moore.constraint.expr
func.func @test_implication_constraint(%obj: !moore.class<@ImplicationConstraintClass>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@ImplicationConstraintClass>
  return %success : i1
}

/// Test class with ConstraintIfElseOp
/// Corresponds to SystemVerilog: constraint c { if (mode) x == 1; else x == 0; }

moore.class.classdecl @IfElseConstraintClass {
  moore.class.propertydecl @mode : !moore.i1 rand_mode rand
  moore.class.propertydecl @x : !moore.i32 rand_mode rand
  moore.constraint.block @ifelse_c {
  ^bb0(%mode: !moore.i1, %x: !moore.i32):
    %c0 = moore.constant 0 : i32
    %c1 = moore.constant 1 : i32
    moore.constraint.if_else %mode : i1 {
      %x_eq_1 = moore.eq %x, %c1 : i32 -> i1
      moore.constraint.expr %x_eq_1 : i1
    } else {
      %x_eq_0 = moore.eq %x, %c0 : i32 -> i1
      moore.constraint.expr %x_eq_0 : i1
    }
  }
}

// CHECK-LABEL: func.func @test_ifelse_constraint
// CHECK-NOT: moore.constraint.block
// CHECK-NOT: moore.constraint.if_else
// CHECK-NOT: moore.constraint.expr
func.func @test_ifelse_constraint(%obj: !moore.class<@IfElseConstraintClass>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@IfElseConstraintClass>
  return %success : i1
}

/// Test soft constraint (soft inside)
/// Corresponds to SystemVerilog: constraint c { soft x inside {[0:10]}; }

moore.class.classdecl @SoftConstraintClass {
  moore.class.propertydecl @x : !moore.i32 rand_mode rand
  moore.constraint.block @soft_c {
  ^bb0(%x: !moore.i32):
    moore.constraint.inside %x, [0, 10] : !moore.i32 soft
  }
}

// CHECK-LABEL: func.func @test_soft_constraint
// CHECK-NOT: moore.constraint.block
// CHECK-NOT: moore.constraint.inside
func.func @test_soft_constraint(%obj: !moore.class<@SoftConstraintClass>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@SoftConstraintClass>
  return %success : i1
}

/// Test unique constraint
/// Corresponds to SystemVerilog: constraint c { unique {arr}; }

moore.class.classdecl @UniqueConstraintClass {
  moore.class.propertydecl @arr : !moore.uarray<8 x i32> rand_mode rand
  moore.constraint.block @unique_c {
  ^bb0(%arr: !moore.uarray<8 x i32>):
    moore.constraint.unique %arr : !moore.uarray<8 x i32>
  }
}

// CHECK-LABEL: func.func @test_unique_constraint
// CHECK-NOT: moore.constraint.block
// CHECK-NOT: moore.constraint.unique
func.func @test_unique_constraint(%obj: !moore.class<@UniqueConstraintClass>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@UniqueConstraintClass>
  return %success : i1
}

/// Test solve before constraint
/// Corresponds to SystemVerilog: constraint c { solve a before b; }

moore.class.classdecl @SolveBeforeClass {
  moore.class.propertydecl @a : !moore.i32 rand_mode rand
  moore.class.propertydecl @b : !moore.i32 rand_mode rand
  moore.constraint.block @solve_c {
    moore.constraint.solve_before [@a], [@b]
  }
}

// CHECK-LABEL: func.func @test_solve_before
// CHECK-NOT: moore.constraint.block
// CHECK-NOT: moore.constraint.solve_before
func.func @test_solve_before(%obj: !moore.class<@SolveBeforeClass>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@SolveBeforeClass>
  return %success : i1
}

/// Test multi-range constraint
/// Corresponds to SystemVerilog: constraint c { x inside {1, [5:10], 20}; }

moore.class.classdecl @MultiRangeClass {
  moore.class.propertydecl @x : !moore.i8 rand_mode rand
  moore.constraint.block @multi_c {
  ^bb0(%x: !moore.i8):
    // Ranges: [1,1], [5,10], [20,20] (single values as degenerate ranges)
    moore.constraint.inside %x, [1, 1, 5, 10, 20, 20] : !moore.i8
  }
}

// CHECK-LABEL: func.func @test_multi_range
// CHECK: llvm.call @__moore_randomize_basic
// CHECK: arith.trunci {{.*}} : i32 to i1
// CHECK: return
func.func @test_multi_range(%obj: !moore.class<@MultiRangeClass>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@MultiRangeClass>
  return %success : i1
}

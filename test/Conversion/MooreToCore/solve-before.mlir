// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32
// CHECK-DAG: llvm.func @__moore_randomize_with_range(i64, i64) -> i64
// CHECK-DAG: llvm.func @__moore_is_rand_enabled(!llvm.ptr, !llvm.ptr) -> i32

//===----------------------------------------------------------------------===//
// Solve-Before Constraint Ordering Tests
//===----------------------------------------------------------------------===//
//
// These tests verify that solve-before constraints correctly order the
// application of range constraints during randomization.
// IEEE 1800-2017 Section 18.5.10 "Constraint ordering"
//
// The solve-before constraint `solve a before b` ensures that variable `a`
// is randomized first, and then `b` is randomized, potentially using `a`'s
// value in its constraint evaluation.
//
//===----------------------------------------------------------------------===//

/// Test basic solve-before ordering with two variables
/// SystemVerilog: solve mode before data;
/// Expected: mode constraint should be applied before data constraint

moore.class.classdecl @BasicSolveBefore {
  moore.class.propertydecl @mode : !moore.i8 rand_mode rand
  moore.class.propertydecl @data : !moore.i8 rand_mode rand
  moore.constraint.block @c_order {
    moore.constraint.solve_before [@mode], [@data]
  }
  moore.constraint.block @c_mode {
  ^bb0(%mode: !moore.i8, %data: !moore.i8):
    // mode constrained to [0, 3]
    moore.constraint.inside %mode, [0, 3] : !moore.i8
  }
  moore.constraint.block @c_data {
  ^bb0(%mode: !moore.i8, %data: !moore.i8):
    // data constrained to [10, 50]
    moore.constraint.inside %data, [10, 50] : !moore.i8
  }
}

// CHECK-LABEL: func.func @test_basic_solve_before
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
// Verify that mode is processed before data by checking the order of
// __moore_randomize_with_range calls. The first call should have mode's
// range [0,3] and the second should have data's range [10,50].
// CHECK: llvm.call @__moore_randomize_basic
// First constraint application should be for mode (range [0,3])
// CHECK: llvm.mlir.constant(0 : i64)
// CHECK: llvm.mlir.constant(3 : i64)
// CHECK: llvm.call @__moore_randomize_with_range
// Second constraint application should be for data (range [10,50])
// CHECK: llvm.mlir.constant(10 : i64)
// CHECK: llvm.mlir.constant(50 : i64)
// CHECK: llvm.call @__moore_randomize_with_range
func.func @test_basic_solve_before(%obj: !moore.class<@BasicSolveBefore>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@BasicSolveBefore>
  return %success : i1
}

/// Test solve-before with multiple 'after' variables
/// SystemVerilog: solve mode before data, addr;
/// Expected: mode should be randomized before both data and addr

moore.class.classdecl @SolveBeforeMultiple {
  moore.class.propertydecl @mode : !moore.i8 rand_mode rand
  moore.class.propertydecl @data : !moore.i8 rand_mode rand
  moore.class.propertydecl @addr : !moore.i8 rand_mode rand
  moore.constraint.block @c_order {
    moore.constraint.solve_before [@mode], [@data, @addr]
  }
  moore.constraint.block @c_ranges {
  ^bb0(%mode: !moore.i8, %data: !moore.i8, %addr: !moore.i8):
    moore.constraint.inside %mode, [0, 1] : !moore.i8
    moore.constraint.inside %data, [100, 200] : !moore.i8
    moore.constraint.inside %addr, [0, 255] : !moore.i8
  }
}

// CHECK-LABEL: func.func @test_solve_before_multiple
// CHECK: llvm.call @__moore_randomize_basic
// First constraint should be mode (range [0,1])
// CHECK: llvm.mlir.constant(0 : i64)
// CHECK: llvm.mlir.constant(1 : i64)
// CHECK: llvm.call @__moore_randomize_with_range
func.func @test_solve_before_multiple(%obj: !moore.class<@SolveBeforeMultiple>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@SolveBeforeMultiple>
  return %success : i1
}

/// Test chained solve-before: solve a before b; solve b before c;
/// Expected order: a, then b, then c

moore.class.classdecl @ChainedSolveBefore {
  moore.class.propertydecl @a : !moore.i8 rand_mode rand
  moore.class.propertydecl @b : !moore.i8 rand_mode rand
  moore.class.propertydecl @c : !moore.i8 rand_mode rand
  moore.constraint.block @c_order1 {
    moore.constraint.solve_before [@a], [@b]
  }
  moore.constraint.block @c_order2 {
    moore.constraint.solve_before [@b], [@c]
  }
  moore.constraint.block @c_ranges {
  ^bb0(%a: !moore.i8, %b: !moore.i8, %c: !moore.i8):
    moore.constraint.inside %a, [1, 1] : !moore.i8
    moore.constraint.inside %b, [2, 2] : !moore.i8
    moore.constraint.inside %c, [3, 3] : !moore.i8
  }
}

// CHECK-LABEL: func.func @test_chained_solve_before
// CHECK: llvm.call @__moore_randomize_basic
// First should be a (value 1)
// CHECK: llvm.mlir.constant(1 : i64)
// CHECK: llvm.mlir.constant(1 : i64)
// CHECK: llvm.call @__moore_randomize_with_range
// Second should be b (value 2)
// CHECK: llvm.mlir.constant(2 : i64)
// CHECK: llvm.mlir.constant(2 : i64)
// CHECK: llvm.call @__moore_randomize_with_range
// Third should be c (value 3)
// CHECK: llvm.mlir.constant(3 : i64)
// CHECK: llvm.mlir.constant(3 : i64)
// CHECK: llvm.call @__moore_randomize_with_range
func.func @test_chained_solve_before(%obj: !moore.class<@ChainedSolveBefore>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@ChainedSolveBefore>
  return %success : i1
}

/// Test solve-before with unconstrained variable (no effect on unconstrained)
/// SystemVerilog: solve mode before data; (but only mode has a constraint)

moore.class.classdecl @PartialSolveBefore {
  moore.class.propertydecl @mode : !moore.i8 rand_mode rand
  moore.class.propertydecl @data : !moore.i8 rand_mode rand
  moore.constraint.block @c_order {
    moore.constraint.solve_before [@mode], [@data]
  }
  moore.constraint.block @c_mode {
  ^bb0(%mode: !moore.i8, %data: !moore.i8):
    // Only mode is constrained
    moore.constraint.inside %mode, [5, 10] : !moore.i8
  }
}

// CHECK-LABEL: func.func @test_partial_solve_before
// CHECK: llvm.call @__moore_randomize_basic
// Only mode should have a constraint applied
// CHECK: llvm.mlir.constant(5 : i64)
// CHECK: llvm.mlir.constant(10 : i64)
// CHECK: llvm.call @__moore_randomize_with_range
// No second __moore_randomize_with_range call since data has no constraint
// CHECK: hw.constant true
func.func @test_partial_solve_before(%obj: !moore.class<@PartialSolveBefore>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@PartialSolveBefore>
  return %success : i1
}

/// Test that solve-before ops are erased after processing
/// Verifies the constraint op is not left in the output

moore.class.classdecl @SolveBeforeErased {
  moore.class.propertydecl @x : !moore.i8 rand_mode rand
  moore.class.propertydecl @y : !moore.i8 rand_mode rand
  moore.constraint.block @c {
    moore.constraint.solve_before [@x], [@y]
  }
}

// CHECK-LABEL: func.func @test_solve_before_erased
// CHECK-NOT: moore.constraint.solve_before
// CHECK-NOT: moore.constraint.block
func.func @test_solve_before_erased(%obj: !moore.class<@SolveBeforeErased>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@SolveBeforeErased>
  return %success : i1
}

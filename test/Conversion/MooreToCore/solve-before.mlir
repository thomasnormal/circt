// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32
// CHECK-DAG: llvm.func @__moore_is_rand_enabled(!llvm.ptr, !llvm.ptr) -> i32

//===----------------------------------------------------------------------===//
// Solve-Before Constraint Ordering Tests
//===----------------------------------------------------------------------===//
//
// These tests verify that solve-before constraints are consumed and erased
// during the MooreToCore conversion.  The current lowering does not generate
// separate per-variable range calls; instead it emits a single
// __moore_randomize_basic call that randomizes all rand fields at once.  The
// solve-before ordering metadata is advisory (IEEE 1800-2017 Section 18.5.10)
// and does not affect the lowered IR structure.
//
//===----------------------------------------------------------------------===//

/// Test basic solve-before ordering with two variables
/// SystemVerilog: solve mode before data;

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
// Save field values and check rand_enabled for each field
// CHECK: llvm.getelementptr %[[OBJ]][0, 2]
// CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}})
// CHECK: llvm.getelementptr %[[OBJ]][0, 3]
// CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}})
// Single randomize_basic call for the whole object
// CHECK: llvm.call @__moore_randomize_basic(%[[OBJ]], {{.*}})
// CHECK: arith.trunci {{.*}} : i32 to i1
// Conditionally restore non-rand fields
// CHECK: scf.if
// CHECK: scf.if
// Post-randomize: check any field is rand-enabled and AND with basic result
// CHECK: arith.ori
// CHECK: arith.ori
// CHECK: arith.andi
// CHECK: return
func.func @test_basic_solve_before(%obj: !moore.class<@BasicSolveBefore>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@BasicSolveBefore>
  return %success : i1
}

/// Test solve-before with multiple 'after' variables
/// SystemVerilog: solve mode before data, addr;

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
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
// Three fields saved and checked
// CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}})
// CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}})
// CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}})
// Single randomize_basic call
// CHECK: llvm.call @__moore_randomize_basic(%[[OBJ]], {{.*}})
// Three conditional restores
// CHECK: scf.if
// CHECK: scf.if
// CHECK: scf.if
// CHECK: return
func.func @test_solve_before_multiple(%obj: !moore.class<@SolveBeforeMultiple>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@SolveBeforeMultiple>
  return %success : i1
}

/// Test chained solve-before: solve a before b; solve b before c;

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
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
// Three fields saved and checked
// CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}})
// CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}})
// CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}})
// Single randomize_basic call
// CHECK: llvm.call @__moore_randomize_basic(%[[OBJ]], {{.*}})
// Three conditional restores
// CHECK: scf.if
// CHECK: scf.if
// CHECK: scf.if
// CHECK: return
func.func @test_chained_solve_before(%obj: !moore.class<@ChainedSolveBefore>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@ChainedSolveBefore>
  return %success : i1
}

/// Test solve-before with unconstrained variable (only mode has a constraint)
/// SystemVerilog: solve mode before data;

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
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
// CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}})
// CHECK: llvm.call @__moore_is_rand_enabled(%[[OBJ]], {{.*}})
// CHECK: llvm.call @__moore_randomize_basic(%[[OBJ]], {{.*}})
// CHECK: scf.if
// CHECK: scf.if
// CHECK: return
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
// CHECK: llvm.call @__moore_randomize_basic
// CHECK: return
func.func @test_solve_before_erased(%obj: !moore.class<@SolveBeforeErased>) -> i1 {
  %success = moore.randomize %obj : !moore.class<@SolveBeforeErased>
  return %success : i1
}

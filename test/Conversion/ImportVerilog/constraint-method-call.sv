// RUN: circt-verilog %s --ir-moore | FileCheck %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

//===----------------------------------------------------------------------===//
// Constraint Method Call Tests
// Tests for function/method calls inside constraint blocks.
// IEEE 1800-2017 Section 18.5.12 "Functions in constraints"
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Test 1: Basic function call in constraint
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @basic_func_constraint {
// CHECK:   moore.class.propertydecl @b1 : !moore.i32 rand_mode rand
// CHECK:   moore.class.propertydecl @b2 : !moore.i32 rand_mode rand
// CHECK:   moore.constraint.block @c1 {
// CHECK:     ^bb0(%arg0: !moore.class<@basic_func_constraint>):
// CHECK:       moore.constraint.expr %{{.*}} : i1
// CHECK:   }
// CHECK:   moore.constraint.block @c2 {
// CHECK:     ^bb0(%arg0: !moore.class<@basic_func_constraint>):
// CHECK:       %[[CALL:.*]] = moore.constraint.method_call @"basic_func_constraint::F"(%arg0, %{{.*}}) : (!moore.class<@basic_func_constraint>, !moore.i32) -> !moore.i32
// CHECK:       moore.constraint.expr %{{.*}} : i1
// CHECK:   }
// CHECK: }

class basic_func_constraint;
  rand int b1, b2;

  function int F(input int d);
    F = d;
  endfunction

  constraint c1 { b1 == 5; }
  constraint c2 { b2 == F(b1); }
endclass

//===----------------------------------------------------------------------===//
// Test 2: Method call with multiple arguments
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @multi_arg_method {
// CHECK:   moore.class.propertydecl @x : !moore.i32 rand_mode rand
// CHECK:   moore.class.propertydecl @y : !moore.i32 rand_mode rand
// CHECK:   moore.class.propertydecl @z : !moore.i32 rand_mode rand
// CHECK:   moore.constraint.block @c {
// CHECK:     ^bb0(%arg0: !moore.class<@multi_arg_method>):
// CHECK:       %[[CALL:.*]] = moore.constraint.method_call @"multi_arg_method::sum"(%arg0, %{{.*}}, %{{.*}}) : (!moore.class<@multi_arg_method>, !moore.i32, !moore.i32) -> !moore.i32
// CHECK:       moore.constraint.expr %{{.*}} : i1
// CHECK:   }
// CHECK: }

class multi_arg_method;
  rand int x, y, z;

  function int sum(input int a, input int b);
    return a + b;
  endfunction

  // z must equal the sum of x and y
  constraint c { z == sum(x, y); }
endclass

//===----------------------------------------------------------------------===//
// Test 3: Constraint block with this argument (non-static)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @constraint_with_this {
// CHECK:   moore.class.propertydecl @value : !moore.i32 rand_mode rand
// CHECK:   moore.class.propertydecl @threshold : !moore.i32
// CHECK:   moore.constraint.block @c_value {
// CHECK:     ^bb0(%arg0: !moore.class<@constraint_with_this>):
// CHECK:       %[[PROP:.*]] = moore.class.property_ref %arg0[@threshold]
// CHECK:       %[[READ:.*]] = moore.read %[[PROP]]
// CHECK:       moore.constraint.expr %{{.*}} : i1
// CHECK:   }
// CHECK: }

class constraint_with_this;
  rand int value;
  int threshold;

  function new();
    threshold = 100;
  endfunction

  // Constraint accesses non-rand property through implicit this
  constraint c_value { value < threshold; }
endclass

//===----------------------------------------------------------------------===//
// Test 4: Method call in implication constraint
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @method_in_implication {
// CHECK:   moore.class.propertydecl @enable : !moore.i1 rand_mode rand
// CHECK:   moore.class.propertydecl @data : !moore.i32 rand_mode rand
// CHECK:   moore.constraint.block @c {
// CHECK:     ^bb0(%arg0: !moore.class<@method_in_implication>):
// CHECK:       moore.constraint.implication %{{.*}} : i1 {
// Note: slang constant-folds get_max() to 1000, so no method call in IR
// CHECK:         moore.constant 1000
// CHECK:         moore.constraint.expr %{{.*}} : i1
// CHECK:       }
// CHECK:   }
// CHECK: }

class method_in_implication;
  rand bit enable;
  rand int data;

  function int get_max();
    return 1000;
  endfunction

  // When enabled, data must be less than get_max()
  // Note: slang constant-folds the pure function call
  constraint c { enable -> (data < get_max()); }
endclass

//===----------------------------------------------------------------------===//
// Test 5: Method call in if-else constraint
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @method_in_ifelse {
// CHECK:   moore.class.propertydecl @mode : !moore.i1 rand_mode rand
// CHECK:   moore.class.propertydecl @value : !moore.i32 rand_mode rand
// CHECK:   moore.constraint.block @c {
// CHECK:     ^bb0(%arg0: !moore.class<@method_in_ifelse>):
// CHECK:       moore.constraint.if_else %{{.*}} : i1 {
// Note: slang constant-folds high_bound() to 1000
// CHECK:         moore.constant 1000
// CHECK:         moore.constraint.expr %{{.*}} : i1
// CHECK:       } else {
// Note: slang constant-folds low_bound() to 100
// CHECK:         moore.constant 100
// CHECK:         moore.constraint.expr %{{.*}} : i1
// CHECK:       }
// CHECK:   }
// CHECK: }

class method_in_ifelse;
  rand bit mode;
  rand int value;

  function int high_bound();
    return 1000;
  endfunction

  function int low_bound();
    return 100;
  endfunction

  // Note: slang constant-folds the pure function calls
  constraint c {
    if (mode) {
      value < high_bound();
    } else {
      value < low_bound();
    }
  }
endclass

//===----------------------------------------------------------------------===//
// Test module instantiating all constraint classes
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @test_constraint_method_call
module test_constraint_method_call;
  initial begin
    automatic basic_func_constraint t1 = new();
    automatic multi_arg_method t2 = new();
    automatic constraint_with_this t3 = new();
    automatic method_in_implication t4 = new();
    automatic method_in_ifelse t5 = new();

    $display("Test constraint method call IR generation complete");
  end
endmodule

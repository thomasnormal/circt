// RUN: circt-verilog %s --ir-moore | FileCheck %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

//===----------------------------------------------------------------------===//
// Constraint Solving Tests - Iteration 52 Track B
// Tests for various constraint types and their lowering to Moore IR.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Test 1: Basic range constraints with inside
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @constrained_class {
// CHECK:   moore.class.propertydecl @value : !moore.i8 rand_mode rand
// CHECK:   moore.class.propertydecl @small_val : !moore.i4 rand_mode rand
// CHECK:   moore.constraint.block @c1 {
// CHECK:     moore.constraint.expr {{.*}} : i1
// CHECK:   }
// CHECK:   moore.constraint.block @c2 {
// CHECK:     moore.constraint.expr {{.*}} : i1
// CHECK:   }
// CHECK: }

class constrained_class;
  rand bit [7:0] value;
  rand bit [3:0] small_val;

  constraint c1 { value inside {[10:200]}; }
  constraint c2 { small_val inside {[1:5]}; }

  function new();
    value = 0;
    small_val = 0;
  endfunction
endclass

//===----------------------------------------------------------------------===//
// Test 2: Expression constraints (greater than, less than)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @expr_constrained {
// CHECK:   moore.class.propertydecl @x : !moore.i32 rand_mode rand
// CHECK:   moore.constraint.block @c_range {
// CHECK:     %[[GT:.*]] = moore.sgt {{.*}} : i32 -> i1
// CHECK:     moore.constraint.expr %[[GT]] : i1
// CHECK:     %[[LT:.*]] = moore.slt {{.*}} : i32 -> i1
// CHECK:     moore.constraint.expr %[[LT]] : i1
// CHECK:   }
// CHECK: }

class expr_constrained;
  rand int x;
  constraint c_range { x > 10; x < 200; }
endclass

//===----------------------------------------------------------------------===//
// Test 3: Multiple value ranges with inside
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @multi_range_class {
// CHECK:   moore.class.propertydecl @data : !moore.i8 rand_mode rand
// CHECK:   moore.constraint.block @c_multi {
// CHECK:     moore.constraint.expr {{.*}} : i1
// CHECK:   }
// CHECK: }

class multi_range_class;
  rand bit [7:0] data;
  // Multiple ranges: 1-5, 10-20, single value 100
  constraint c_multi { data inside {[1:5], [10:20], 100}; }
endclass

//===----------------------------------------------------------------------===//
// Test 4: Soft constraints
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @soft_constrained {
// CHECK:   moore.class.propertydecl @prio : !moore.i8 rand_mode rand
// CHECK:   moore.constraint.block @c_soft {
// CHECK:     %[[EQ:.*]] = moore.eq {{.*}} : i32 -> i1
// CHECK:     moore.constraint.expr %[[EQ]] : i1 soft
// CHECK:   }
// CHECK: }

class soft_constrained;
  rand bit [7:0] prio;
  // Soft constraint provides default value
  constraint c_soft { soft prio == 42; }
endclass

//===----------------------------------------------------------------------===//
// Test 5: Implication constraint
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @implication_class {
// CHECK:   moore.class.propertydecl @mode : !moore.i1 rand_mode rand
// CHECK:   moore.class.propertydecl @speed : !moore.i8 rand_mode rand
// CHECK:   moore.constraint.block @c_impl {
// CHECK:     moore.constraint.implication {{.*}} : i1 {
// CHECK:       moore.constraint.expr {{.*}} : i1
// CHECK:     }
// CHECK:   }
// CHECK: }

class implication_class;
  rand bit mode;
  rand bit [7:0] speed;
  // If mode is fast, speed must be high
  constraint c_impl { mode -> speed inside {[100:255]}; }
endclass

//===----------------------------------------------------------------------===//
// Test 6: If-else constraint
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @ifelse_class {
// CHECK:   moore.class.propertydecl @is_write : !moore.i1 rand_mode rand
// CHECK:   moore.class.propertydecl @addr : !moore.i16 rand_mode rand
// CHECK:   moore.constraint.block @c_ifelse {
// CHECK:     moore.constraint.if_else {{.*}} : i1 {
// CHECK:       moore.constraint.expr {{.*}} : i1
// CHECK:     } else {
// CHECK:       moore.constraint.expr {{.*}} : i1
// CHECK:     }
// CHECK:   }
// CHECK: }

class ifelse_class;
  rand bit is_write;
  rand bit [15:0] addr;
  // Different address ranges for read vs write
  constraint c_ifelse {
    if (is_write)
      addr inside {[0:1023]};
    else
      addr inside {[1024:2047]};
  }
endclass

//===----------------------------------------------------------------------===//
// Test 7: Unique constraint with array
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @unique_class {
// CHECK:   moore.class.propertydecl @arr : !moore.uarray<4 x i8> rand_mode rand
// CHECK:   moore.constraint.block @c_unique {
// CHECK:     moore.constraint.unique {{.*}} : !moore.uarray<4 x i8>
// CHECK:   }
// CHECK: }

class unique_class;
  rand bit [7:0] arr[4];
  // All array elements must be unique
  constraint c_unique { unique {arr}; }
endclass

//===----------------------------------------------------------------------===//
// Test 8: Solve...before constraint
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @solve_before_class {
// CHECK:   moore.class.propertydecl @mode : !moore.i2 rand_mode rand
// CHECK:   moore.class.propertydecl @data : !moore.i8 rand_mode rand
// CHECK:   moore.class.propertydecl @addr : !moore.i16 rand_mode rand
// CHECK:   moore.constraint.block @c_solve {
// CHECK:     moore.constraint.solve_before [@mode], [@data, @addr]
// CHECK:   }
// CHECK: }

class solve_before_class;
  rand bit [1:0] mode;
  rand bit [7:0] data;
  rand bit [15:0] addr;
  // Solve mode first, then constrain data and addr based on mode
  constraint c_solve {
    solve mode before data, addr;
  }
endclass

//===----------------------------------------------------------------------===//
// Test 9: Solve...before with conditional constraints
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @solve_conditional_class {
// CHECK:   moore.class.propertydecl @is_write : !moore.i1 rand_mode rand
// CHECK:   moore.class.propertydecl @size : !moore.i8 rand_mode rand
// CHECK:   moore.constraint.block @c_order {
// CHECK:     moore.constraint.solve_before [@is_write], [@size]
// CHECK:   }
// CHECK:   moore.constraint.block @c_size {
// CHECK:     moore.constraint.if_else
// CHECK:   }
// CHECK: }

class solve_conditional_class;
  rand bit is_write;
  rand bit [7:0] size;
  // Ensure is_write is solved before size
  constraint c_order { solve is_write before size; }
  // Size depends on operation type
  constraint c_size {
    if (is_write)
      size inside {[1:64]};
    else
      size inside {[1:256]};
  }
endclass

//===----------------------------------------------------------------------===//
// Test 10: Unique constraint with multiple individual variables
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @unique_vars_class {
// CHECK:   moore.class.propertydecl @a : !moore.i8 rand_mode rand
// CHECK:   moore.class.propertydecl @b : !moore.i8 rand_mode rand
// CHECK:   moore.class.propertydecl @c : !moore.i8 rand_mode rand
// CHECK:   moore.constraint.block @c_unique_vars {
// CHECK:     moore.constraint.unique {{.*}} : !moore.i8, !moore.i8, !moore.i8
// CHECK:   }
// CHECK: }

class unique_vars_class;
  rand bit [7:0] a;
  rand bit [7:0] b;
  rand bit [7:0] c;
  // All three variables must have different values
  constraint c_unique_vars { unique {a, b, c}; }
endclass

//===----------------------------------------------------------------------===//
// Test 11: Combined constraints in a single block
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @combined_class {
// CHECK:   moore.class.propertydecl @a : !moore.i8 rand_mode rand
// CHECK:   moore.class.propertydecl @b : !moore.i8 rand_mode rand
// CHECK:   moore.constraint.block @c_combined {
// CHECK:     moore.constraint.expr {{.*}} : i1
// CHECK:     moore.constraint.expr {{.*}} : i1
// CHECK:     %[[LT:.*]] = moore.ult {{.*}} : i8 -> i1
// CHECK:     moore.constraint.expr %[[LT]] : i1
// CHECK:   }
// CHECK: }

class combined_class;
  rand bit [7:0] a;
  rand bit [7:0] b;
  // Multiple constraints in one block
  constraint c_combined {
    a inside {[1:100]};
    b inside {[1:100]};
    a < b;
  }
endclass

//===----------------------------------------------------------------------===//
// Test 12: Module instantiating constrained classes
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.module @test_constraints
module test_constraints;
  initial begin
    constrained_class obj1;
    expr_constrained obj2;
    multi_range_class obj3;
    soft_constrained obj4;
    implication_class obj5;
    ifelse_class obj6;
    unique_class obj7;
    combined_class obj8;
    solve_before_class obj9;
    solve_conditional_class obj10;
    unique_vars_class obj11;
    int success;

    obj1 = new();
    obj2 = new();
    obj3 = new();
    obj4 = new();
    obj5 = new();
    obj6 = new();
    obj7 = new();
    obj8 = new();
    obj9 = new();
    obj10 = new();
    obj11 = new();

    // CHECK: moore.randomize
    success = obj1.randomize();
    $display("obj1.value=%0d (10-200), obj1.small_val=%0d (1-5)", obj1.value, obj1.small_val);

    // CHECK: moore.randomize
    success = obj2.randomize();
    $display("obj2.x=%0d (10 < x < 200)", obj2.x);

    // CHECK: moore.randomize
    success = obj3.randomize();
    $display("obj3.data=%0d (1-5, 10-20, or 100)", obj3.data);

    // CHECK: moore.randomize
    success = obj4.randomize();
    $display("obj4.prio=%0d (soft default 42)", obj4.prio);

    // CHECK: moore.randomize
    success = obj5.randomize();
    $display("obj5.mode=%0d, obj5.speed=%0d", obj5.mode, obj5.speed);

    // CHECK: moore.randomize
    success = obj6.randomize();
    $display("obj6.is_write=%0d, obj6.addr=%0d", obj6.is_write, obj6.addr);

    // CHECK: moore.randomize
    success = obj7.randomize();
    $display("obj7.arr unique values");

    // CHECK: moore.randomize
    success = obj8.randomize();
    $display("obj8.a=%0d, obj8.b=%0d (a < b)", obj8.a, obj8.b);

    // CHECK: moore.randomize
    success = obj9.randomize();
    $display("obj9.mode=%0d, data=%0d, addr=%0d (mode solved first)", obj9.mode, obj9.data, obj9.addr);

    // CHECK: moore.randomize
    success = obj10.randomize();
    $display("obj10.is_write=%0d, size=%0d (is_write solved first)", obj10.is_write, obj10.size);

    // CHECK: moore.randomize
    success = obj11.randomize();
    $display("obj11.a=%0d, b=%0d, c=%0d (all unique)", obj11.a, obj11.b, obj11.c);
  end
endmodule

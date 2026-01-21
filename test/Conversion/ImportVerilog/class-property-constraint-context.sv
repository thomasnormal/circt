// RUN: circt-translate --import-verilog %s 2>&1 | FileCheck %s
// REQUIRES: slang

// Test case for static class property resolution fix.
// Non-static properties in constraint blocks should not be treated as static.

// Test 1: Property initializer that references another property
// During class body conversion, thisRef is cleared, so the initializer
// of 'b' which references 'a' is converted without an implicit 'this'.

// CHECK-LABEL: moore.class.classdecl @InitBugTest
// CHECK:   moore.class.propertydecl @a : !moore.i32
// CHECK:   moore.class.propertydecl @b : !moore.i32
// CHECK: }

class InitBugTest;
    int a;
    int b = a;  // Initializer references 'a' without explicit this
endclass

// Test 2: Constraint that references a property
// During constraint conversion, there's no implicit 'this', so property
// references should create symbolic placeholder variables (not static).

// CHECK-LABEL: moore.class.classdecl @ConstraintBugTest
// CHECK:   moore.class.propertydecl @x : !moore.i32 rand_mode rand
// CHECK:   moore.class.propertydecl @y : !moore.i32 rand_mode rand
// CHECK:   moore.constraint.block @c {
// Properties are referenced as placeholder variables in constraints
// CHECK:     %y = moore.variable : <i32>
// CHECK:     %x = moore.variable : <i32>
// CHECK:   }
// CHECK: }

class ConstraintBugTest;
    rand int x;
    rand int y;
    constraint c { y == x; }  // Accessing x and y within constraint
endclass

// Verify no static property warnings are emitted
// CHECK-NOT: warning: static class property

// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

//===----------------------------------------------------------------------===//
// In-line Constraint Checker (randomize(null))
// IEEE 1800-2017 Section 18.11.1
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @ConstraintChecker {
// CHECK:   moore.class.propertydecl @x : !moore.i32 rand_mode rand
// CHECK:   moore.class.propertydecl @v : !moore.i32
// CHECK:   moore.constraint.block @c1 {
// CHECK:   }
// CHECK: }

class ConstraintChecker;
    rand int x;
    int v;
    constraint c1 { x < v; };
endclass

/// Check that randomize(null) generates check_only mode
/// No pre_randomize or post_randomize should be called

// CHECK-LABEL: moore.module @testRandomizeNull() {
// CHECK:   %obj = moore.variable : <class<@ConstraintChecker>>
// CHECK:   %ret = moore.variable : <i32>
// CHECK:   moore.procedure initial {
// CHECK:     %{{.+}} = moore.class.new : <@ConstraintChecker>
// CHECK:     moore.blocking_assign %obj
// CHECK:     moore.read %obj
// CHECK-NOT: moore.call_pre_randomize
// CHECK:     moore.randomize %{{.+}} check_only : <@ConstraintChecker>
// CHECK-NOT: moore.call_post_randomize
// CHECK:     moore.conversion
// CHECK:     moore.blocking_assign %ret
// CHECK:     moore.return
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module testRandomizeNull;
    ConstraintChecker obj;
    int ret;
    initial begin
        obj = new;
        obj.x = 2;
        obj.v = 1;
        ret = obj.randomize(null);
    end
endmodule

//===----------------------------------------------------------------------===//
// In-line Random Variable Control (randomize(v, w))
// IEEE 1800-2017 Section 18.11
//===----------------------------------------------------------------------===//

// CHECK-LABEL: moore.class.classdecl @VariableControl {
// CHECK:   moore.class.propertydecl @x : !moore.i32 rand_mode rand
// CHECK:   moore.class.propertydecl @y : !moore.i32 rand_mode rand
// CHECK:   moore.class.propertydecl @v : !moore.i32
// CHECK:   moore.class.propertydecl @w : !moore.i32
// CHECK:   moore.constraint.block @c {
// CHECK:   }
// CHECK: }

class VariableControl;
    rand int x, y;
    int v, w;
    constraint c { x < v && y > w; };
endclass

/// Check that randomize(v, w) generates variable_list attribute
/// This makes v and w the random variables instead of x and y

// CHECK-LABEL: moore.module @testRandomizeVarList() {
// CHECK:   %obj = moore.variable : <class<@VariableControl>>
// CHECK:   moore.procedure initial {
// CHECK:     %{{.+}} = moore.class.new : <@VariableControl>
// CHECK:     moore.blocking_assign %obj
// CHECK:     moore.read %obj
// CHECK:     moore.call_pre_randomize %{{.+}} : <@VariableControl>
// CHECK:     moore.randomize %{{.+}} variable_list([@v, @w]) : <@VariableControl>
// CHECK:     moore.call_post_randomize %{{.+}} : <@VariableControl>
// CHECK:     moore.return
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module testRandomizeVarList;
    VariableControl obj;
    initial begin
        obj = new;
        obj.randomize(v, w);
    end
endmodule

/// Check that normal randomize() does NOT have check_only or variable_list

// CHECK-LABEL: moore.module @testNormalRandomize() {
// CHECK:   %obj = moore.variable : <class<@ConstraintChecker>>
// CHECK:   %ret = moore.variable : <i32>
// CHECK:   moore.procedure initial {
// CHECK:     %{{.+}} = moore.class.new : <@ConstraintChecker>
// CHECK:     moore.blocking_assign %obj
// CHECK:     moore.read %obj
// CHECK:     moore.call_pre_randomize %{{.+}} : <@ConstraintChecker>
// CHECK:     moore.randomize %{{.+}} : <@ConstraintChecker>
// CHECK:     moore.call_post_randomize %{{.+}} : <@ConstraintChecker>
// CHECK:     moore.conversion
// CHECK:     moore.blocking_assign %ret
// CHECK:     moore.return
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module testNormalRandomize;
    ConstraintChecker obj;
    int ret;
    initial begin
        obj = new;
        ret = obj.randomize();
    end
endmodule

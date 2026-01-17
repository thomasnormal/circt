// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

//===----------------------------------------------------------------------===//
// Basic Randomize Tests
//===----------------------------------------------------------------------===//

/// Check basic class with rand properties

// CHECK-LABEL: moore.class.classdecl @Transaction {
// CHECK:   moore.class.propertydecl @data : !moore.i32 rand_mode rand
// CHECK:   moore.class.propertydecl @addr : !moore.i8 rand_mode rand
// CHECK: }

class Transaction;
    rand int data;
    rand bit [7:0] addr;
endclass

/// Check randomize() call on class object

// CHECK-LABEL: moore.module @testBasicRandomize() {
// CHECK:   %t = moore.variable : <class<@Transaction>>
// CHECK:   %success = moore.variable : <i32>
// CHECK:   moore.procedure initial {
// CHECK:     %[[NEW:.+]] = moore.class.new : <@Transaction>
// CHECK:     moore.blocking_assign %t, %[[NEW]] : class<@Transaction>
// CHECK:     %[[OBJ:.+]] = moore.read %t : <class<@Transaction>>
// CHECK:     %[[RAND_RESULT:.+]] = moore.randomize %[[OBJ]] : <@Transaction>
// CHECK:     %[[CONV:.+]] = moore.conversion %[[RAND_RESULT]] : i1 -> !moore.i32
// CHECK:     moore.blocking_assign %success, %[[CONV]] : i32
// CHECK:     moore.return
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module testBasicRandomize;
    Transaction t;
    int success;
    initial begin
        t = new;
        success = t.randomize();
    end
endmodule

/// Check randomize() with constraint block

// CHECK-LABEL: moore.class.classdecl @ConstrainedTransaction {
// CHECK:   moore.class.propertydecl @x : !moore.i32 rand_mode rand
// CHECK:   moore.constraint.block @valid_range {
// CHECK:     %[[GT:.*]] = moore.sgt {{.*}} : i32 -> i1
// CHECK:     moore.constraint.expr %[[GT]] : i1
// CHECK:     %[[LT:.*]] = moore.slt {{.*}} : i32 -> i1
// CHECK:     moore.constraint.expr %[[LT]] : i1
// CHECK:   }
// CHECK: }

class ConstrainedTransaction;
    rand int x;
    constraint valid_range { x > 0; x < 100; }
endclass

// CHECK-LABEL: moore.module @testConstrainedRandomize() {
// CHECK:   %t = moore.variable : <class<@ConstrainedTransaction>>
// CHECK:   %success = moore.variable : <i32>
// CHECK:   moore.procedure initial {
// CHECK:     %[[NEW:.+]] = moore.class.new : <@ConstrainedTransaction>
// CHECK:     moore.blocking_assign %t, %[[NEW]] : class<@ConstrainedTransaction>
// CHECK:     %[[OBJ:.+]] = moore.read %t : <class<@ConstrainedTransaction>>
// CHECK:     %[[RAND_RESULT:.+]] = moore.randomize %[[OBJ]] : <@ConstrainedTransaction>
// CHECK:     %[[CONV:.+]] = moore.conversion %[[RAND_RESULT]] : i1 -> !moore.i32
// CHECK:     moore.blocking_assign %success, %[[CONV]] : i32
// CHECK:     moore.return
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module testConstrainedRandomize;
    ConstrainedTransaction t;
    int success;
    initial begin
        t = new;
        success = t.randomize();
    end
endmodule

/// Check randomize() with randc property

// CHECK-LABEL: moore.class.classdecl @CyclicTransaction {
// CHECK:   moore.class.propertydecl @id : !moore.i8 rand_mode randc
// CHECK: }

class CyclicTransaction;
    randc bit [7:0] id;
endclass

// CHECK-LABEL: moore.module @testCyclicRandomize() {
// CHECK:   %t = moore.variable : <class<@CyclicTransaction>>
// CHECK:   %success = moore.variable : <i32>
// CHECK:   moore.procedure initial {
// CHECK:     %[[NEW:.+]] = moore.class.new : <@CyclicTransaction>
// CHECK:     moore.blocking_assign %t, %[[NEW]] : class<@CyclicTransaction>
// CHECK:     %[[OBJ:.+]] = moore.read %t : <class<@CyclicTransaction>>
// CHECK:     %[[RAND_RESULT:.+]] = moore.randomize %[[OBJ]] : <@CyclicTransaction>
// CHECK:     %[[CONV:.+]] = moore.conversion %[[RAND_RESULT]] : i1 -> !moore.i32
// CHECK:     moore.blocking_assign %success, %[[CONV]] : i32
// CHECK:     moore.return
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module testCyclicRandomize;
    CyclicTransaction t;
    int success;
    initial begin
        t = new;
        success = t.randomize();
    end
endmodule

/// Check randomize() in conditional context

// CHECK-LABEL: moore.module @testRandomizeConditional() {
// CHECK:   %t = moore.variable : <class<@Transaction>>
// CHECK:   %count = moore.variable : <i32>
// CHECK:   moore.procedure initial {
// CHECK:     %[[NEW:.+]] = moore.class.new : <@Transaction>
// CHECK:     moore.blocking_assign %t, %[[NEW]] : class<@Transaction>
// CHECK:     %[[OBJ:.+]] = moore.read %t : <class<@Transaction>>
// CHECK:     %[[RAND_RESULT:.+]] = moore.randomize %[[OBJ]] : <@Transaction>
// CHECK:     %[[CONV:.+]] = moore.conversion %[[RAND_RESULT]] : i1 -> !moore.i32
// CHECK:     %[[BOOL:.+]] = moore.bool_cast %[[CONV]] : i32 -> i1
// CHECK:     %[[NEGBOOL:.+]] = moore.not %[[BOOL]] : i1
// CHECK:     moore.return
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module testRandomizeConditional;
    Transaction t;
    int count;
    initial begin
        t = new;
        if (!t.randomize()) begin
            count = count + 1;
        end
    end
endmodule

/// Check randomize() call inside a class method

// CHECK-LABEL: moore.class.classdecl @SelfRandomizer {
// CHECK:   moore.class.propertydecl @value : !moore.i32 rand_mode rand
// CHECK: }
// CHECK: func.func private @"SelfRandomizer::doRandomize"(%arg0: !moore.class<@SelfRandomizer>) -> !moore.i32 {
// CHECK:   %[[RAND_RESULT:.+]] = moore.randomize %arg0 : <@SelfRandomizer>
// CHECK:   %[[CONV:.+]] = moore.conversion %[[RAND_RESULT]] : i1 -> !moore.i32
// CHECK:   return %[[CONV]] : !moore.i32
// CHECK: }

class SelfRandomizer;
    rand int value;

    function int doRandomize();
        return this.randomize();
    endfunction
endclass

// CHECK-LABEL: moore.module @testMethodRandomize() {
// CHECK:   %t = moore.variable : <class<@SelfRandomizer>>
// CHECK:   %success = moore.variable : <i32>
// CHECK:   moore.procedure initial {
// CHECK:     %[[NEW:.+]] = moore.class.new : <@SelfRandomizer>
// CHECK:     moore.blocking_assign %t, %[[NEW]] : class<@SelfRandomizer>
// CHECK:     %[[OBJ:.+]] = moore.read %t : <class<@SelfRandomizer>>
// CHECK:     %[[CALL_RET:.+]] = func.call @"SelfRandomizer::doRandomize"(%[[OBJ]]) : (!moore.class<@SelfRandomizer>) -> !moore.i32
// CHECK:     moore.blocking_assign %success, %[[CALL_RET]] : i32
// CHECK:     moore.return
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module testMethodRandomize;
    SelfRandomizer t;
    int success;
    initial begin
        t = new;
        success = t.doRandomize();
    end
endmodule

//===----------------------------------------------------------------------===//
// Inline Constraint Tests (with clause)
//===----------------------------------------------------------------------===//

/// Check randomize() with simple inline constraint

// CHECK-LABEL: moore.class.classdecl @InlineConstraintTx {
// CHECK:   moore.class.propertydecl @x : !moore.i32 rand_mode rand
// CHECK:   moore.class.propertydecl @y : !moore.i32 rand_mode rand
// CHECK: }

class InlineConstraintTx;
    rand int x;
    rand int y;
endclass

/// Test basic inline constraint: obj.randomize() with { x > 0; }
/// The inline constraint is captured in the randomize op's region.

// CHECK-LABEL: moore.module @testInlineConstraint() {
// CHECK:   %t = moore.variable : <class<@InlineConstraintTx>>
// CHECK:   %success = moore.variable : <i32>
// CHECK:   moore.procedure initial {
// CHECK:     %[[NEW:.+]] = moore.class.new : <@InlineConstraintTx>
// CHECK:     moore.blocking_assign %t, %[[NEW]] : class<@InlineConstraintTx>
// CHECK:     %[[OBJ:.+]] = moore.read %t : <class<@InlineConstraintTx>>
// CHECK:     %[[RAND_RESULT:.+]] = moore.randomize %[[OBJ]] : <@InlineConstraintTx> {
// CHECK:       %{{.+}} = moore.variable
// CHECK:       %[[XVAL:.+]] = moore.read %{{.+}}
// CHECK:       %[[CONST0:.+]] = moore.constant 0 : i32
// CHECK:       %[[GT:.+]] = moore.sgt %[[XVAL]], %[[CONST0]] : i32 -> i1
// CHECK:       moore.constraint.expr %[[GT]] : i1
// CHECK:     }
// CHECK:     %[[CONV:.+]] = moore.conversion %[[RAND_RESULT]] : i1 -> !moore.i32
// CHECK:     moore.blocking_assign %success, %[[CONV]] : i32
// CHECK:     moore.return
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module testInlineConstraint;
    InlineConstraintTx t;
    int success;
    initial begin
        t = new;
        success = t.randomize() with { x > 0; };
    end
endmodule

/// Test inline constraint with multiple expressions

// CHECK-LABEL: moore.module @testMultipleInlineConstraints() {
// CHECK:   %t = moore.variable : <class<@InlineConstraintTx>>
// CHECK:   %success = moore.variable : <i32>
// CHECK:   moore.procedure initial {
// CHECK:     %[[NEW:.+]] = moore.class.new : <@InlineConstraintTx>
// CHECK:     moore.blocking_assign %t, %[[NEW]] : class<@InlineConstraintTx>
// CHECK:     %[[OBJ:.+]] = moore.read %t : <class<@InlineConstraintTx>>
// CHECK:     %[[RAND_RESULT:.+]] = moore.randomize %[[OBJ]] : <@InlineConstraintTx> {
// CHECK:       %{{.+}} = moore.sgt
// CHECK:       moore.constraint.expr %{{.+}} : i1
// CHECK:       %{{.+}} = moore.slt
// CHECK:       moore.constraint.expr %{{.+}} : i1
// CHECK:     }
// CHECK:     %[[CONV:.+]] = moore.conversion %[[RAND_RESULT]] : i1 -> !moore.i32
// CHECK:     moore.blocking_assign %success, %[[CONV]] : i32
// CHECK:     moore.return
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module testMultipleInlineConstraints;
    InlineConstraintTx t;
    int success;
    initial begin
        t = new;
        success = t.randomize() with { x > 0; x < 100; };
    end
endmodule

/// Test inline constraint with equality

// CHECK-LABEL: moore.module @testInlineConstraintEquality() {
// CHECK:   %t = moore.variable : <class<@InlineConstraintTx>>
// CHECK:   %success = moore.variable : <i32>
// CHECK:   moore.procedure initial {
// CHECK:     %[[NEW:.+]] = moore.class.new : <@InlineConstraintTx>
// CHECK:     moore.blocking_assign %t, %[[NEW]] : class<@InlineConstraintTx>
// CHECK:     %[[OBJ:.+]] = moore.read %t : <class<@InlineConstraintTx>>
// CHECK:     %[[RAND_RESULT:.+]] = moore.randomize %[[OBJ]] : <@InlineConstraintTx> {
// CHECK:       %{{.+}} = moore.eq
// CHECK:       moore.constraint.expr %{{.+}} : i1
// CHECK:     }
// CHECK:   }
// CHECK: }

module testInlineConstraintEquality;
    InlineConstraintTx t;
    int success;
    initial begin
        t = new;
        success = t.randomize() with { x == 42; };
    end
endmodule

/// Test std::randomize() with inline constraints

// CHECK-LABEL: moore.module @testStdRandomizeWithConstraint() {
// CHECK:   %x = moore.variable : <i32>
// CHECK:   %y = moore.variable : <i32>
// CHECK:   %success = moore.variable : <i32>
// CHECK:   moore.procedure initial {
// CHECK:     %[[RAND_RESULT:.+]] = moore.std_randomize %x, %y : !moore.ref<i32>, !moore.ref<i32> {
// CHECK:       %[[XVAL:.+]] = moore.read %x
// CHECK:       %[[YVAL:.+]] = moore.read %y
// CHECK:       %[[LT:.+]] = moore.slt %[[XVAL]], %[[YVAL]] : i32 -> i1
// CHECK:       moore.constraint.expr %[[LT]] : i1
// CHECK:     }
// CHECK:     %[[CONV:.+]] = moore.conversion %[[RAND_RESULT]] : i1 -> !moore.i32
// CHECK:     moore.blocking_assign %success, %[[CONV]] : i32
// CHECK:   }
// CHECK: }

module testStdRandomizeWithConstraint;
    int x, y;
    int success;
    initial begin
        success = std::randomize(x, y) with { x < y; };
    end
endmodule

// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

//===----------------------------------------------------------------------===//
// Test 1: Static property access via instance (obj.static_prop)
//===----------------------------------------------------------------------===//
// SystemVerilog allows accessing static properties through an instance:
//   Foo f;
//   f.static_prop = 1;  // Legal, equivalent to Foo::static_prop = 1
// This should generate GetGlobalVariableOp, not ClassPropertyRefOp.

// CHECK-LABEL: moore.class.classdecl @StaticViaInstance {
// CHECK:   moore.class.propertydecl @instance_prop : !moore.i32
// CHECK: }
// CHECK: moore.global_variable @"StaticViaInstance::static_prop" : !moore.i32

class StaticViaInstance;
    int instance_prop;
    static int static_prop;
endclass

// CHECK-LABEL: moore.module @testStaticViaInstanceRead() {
// CHECK:   %result = moore.variable : <i32>
// CHECK:   moore.procedure initial {
// Access static property through instance - should use GetGlobalVariableOp
// CHECK:     moore.get_global_variable @"StaticViaInstance::static_prop" : <i32>
// CHECK:     moore.read
// CHECK:     moore.blocking_assign %result
// CHECK:     moore.return
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module testStaticViaInstanceRead;
    StaticViaInstance obj;
    int result;
    initial begin
        // Read static property via instance - this is the key fix
        result = obj.static_prop;
    end
endmodule

// CHECK-LABEL: moore.module @testStaticViaInstanceWrite() {
// CHECK:   moore.procedure initial {
// Write to static property through instance - should use GetGlobalVariableOp
// CHECK:     moore.get_global_variable @"StaticViaInstance::static_prop" : <i32>
// CHECK:     moore.blocking_assign
// CHECK:     moore.return
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module testStaticViaInstanceWrite;
    StaticViaInstance obj;
    initial begin
        // Write static property via instance
        obj.static_prop = 42;
    end
endmodule

// Also test that regular static access (Class::prop) still works
// CHECK-LABEL: moore.module @testStaticViaClass() {
// CHECK:   %result = moore.variable : <i32>
// CHECK:   moore.procedure initial {
// CHECK:     moore.get_global_variable @"StaticViaInstance::static_prop" : <i32>
// CHECK:     moore.read
// CHECK:     moore.blocking_assign %result
// CHECK:     moore.return
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module testStaticViaClass;
    int result;
    initial begin
        // Traditional static access via class name
        result = StaticViaInstance::static_prop;
    end
endmodule

//===----------------------------------------------------------------------===//
// Test 2: Time type handling
//===----------------------------------------------------------------------===//
// Time type variables and operations should work correctly.
// This tests that Mem2Reg's getDefaultValue handles TimeType properly.

// CHECK-LABEL: moore.module @testTimeReturn() {
// Time variable - time 0 is computed as constant 0 * timescale
// CHECK:   %t = moore.variable : <time>
// CHECK:   moore.procedure initial {
// CHECK:     moore.blocking_assign %t
// CHECK:     moore.return
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

function time timeReturnFunc();
    return 0;
endfunction

module testTimeReturn;
    time t;
    initial begin
        t = timeReturnFunc();
    end
endmodule

// Test time variable declarations (unused variables may be optimized away)
// CHECK-LABEL: moore.module @testTimeVariable() {
// CHECK:   moore.output
// CHECK: }

module testTimeVariable;
    time t1;
    time t2;
endmodule

//===----------------------------------------------------------------------===//
// Test 3: Parameterized class static properties - unique globals per specialization
//===----------------------------------------------------------------------===//
// Each specialization of a parameterized class needs its own unique global
// variable for static properties. Otherwise, different specializations would
// share the same global, causing incorrect behavior.

// CHECK-LABEL: moore.class.classdecl @ParamClass {
// CHECK:   moore.class.propertydecl @data : !moore.l32
// CHECK: }
// The second specialization gets a different name
// CHECK-LABEL: moore.class.classdecl @ParamClass_0 {
// CHECK:   moore.class.propertydecl @data : !moore.l16
// CHECK: }
// Each specialization gets its own global variable
// CHECK: moore.global_variable @"ParamClass::m_instance" : !moore.class<@ParamClass>
// CHECK: moore.global_variable @"ParamClass_0::m_instance" : !moore.class<@ParamClass_0>

class ParamClass #(int WIDTH = 32);
    logic [WIDTH-1:0] data;
    static ParamClass #(WIDTH) m_instance;

    static function ParamClass #(WIDTH) get();
        if (m_instance == null)
            m_instance = new;
        return m_instance;
    endfunction
endclass

// CHECK-LABEL: moore.module @testParamClassStatic() {
// CHECK:   %obj32 = moore.variable : <class<@ParamClass>>
// CHECK:   %obj16 = moore.variable : <class<@ParamClass_0>>
// CHECK:   moore.procedure initial {
// Access to ParamClass#(32) static (default)
// CHECK:     moore.get_global_variable @"ParamClass::m_instance" : <class<@ParamClass>>
// Access to ParamClass#(16) static - different global!
// CHECK:     moore.get_global_variable @"ParamClass_0::m_instance" : <class<@ParamClass_0>>
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module testParamClassStatic;
    ParamClass #(32) obj32;
    ParamClass #(16) obj16;

    initial begin
        // Access static from different specializations
        obj32 = ParamClass#(32)::m_instance;
        obj16 = ParamClass#(16)::m_instance;
    end
endmodule

//===----------------------------------------------------------------------===//
// Test 4: Abstract class with mixed concrete/pure virtual methods
//===----------------------------------------------------------------------===//
// Virtual classes can have a mix of:
// - Pure virtual methods (no implementation)
// - Concrete virtual methods (have implementation)
// - Regular non-virtual methods
// Previously, this combination caused errors during vtable generation.

// CHECK-LABEL: moore.class.classdecl @AbstractMixed {
// Pure virtual method - now has a stub implementation reference with %this argument
// CHECK:   moore.class.methoddecl @pureMethod -> @"AbstractMixed::pureMethod" : (!moore.class<@AbstractMixed>) -> !moore.i32
// Concrete virtual method - has implementation reference
// CHECK:   moore.class.methoddecl @concreteVirtual -> @"AbstractMixed::concreteVirtual" : (!moore.class<@AbstractMixed>) -> !moore.i32
// CHECK: }

virtual class AbstractMixed;
    // Pure virtual method - no implementation
    pure virtual function int pureMethod();

    // Concrete virtual method - has implementation
    virtual function int concreteVirtual();
        return 42;
    endfunction

    // Non-virtual concrete method
    function int regularMethod();
        return 100;
    endfunction
endclass

// Test that a concrete class implementing the abstract class works
// CHECK-LABEL: moore.class.classdecl @ConcreteMixed extends @AbstractMixed {
// Override of pure virtual
// CHECK:   moore.class.methoddecl @pureMethod -> @"ConcreteMixed::pureMethod" : (!moore.class<@ConcreteMixed>) -> !moore.i32
// Override of concrete virtual
// CHECK:   moore.class.methoddecl @concreteVirtual -> @"ConcreteMixed::concreteVirtual" : (!moore.class<@ConcreteMixed>) -> !moore.i32
// CHECK: }

// Note: vtable declarations are not currently emitted in the Moore dialect output.
// The vtable is accessed at runtime via moore.vtable.load_method.

class ConcreteMixed extends AbstractMixed;
    virtual function int pureMethod();
        return 1;
    endfunction

    virtual function int concreteVirtual();
        return 2;
    endfunction
endclass

// CHECK: func.func private @"ConcreteMixed::pureMethod"
// CHECK:   moore.constant 1 : i32
// CHECK:   return
// CHECK: }

// CHECK: func.func private @"ConcreteMixed::concreteVirtual"
// CHECK:   moore.constant 2 : i32
// CHECK:   return
// CHECK: }

// CHECK-LABEL: moore.module @testAbstractMixed() {
// CHECK:   %obj = moore.variable : <class<@ConcreteMixed>>
// CHECK:   %result = moore.variable : <i32>
// CHECK:   moore.procedure initial {
// CHECK:     moore.class.new : <@ConcreteMixed>
// CHECK:     moore.blocking_assign %obj
// Virtual dispatch for pureMethod
// CHECK:     moore.read %obj
// CHECK:     moore.vtable.load_method {{%.+}} : @pureMethod of <@ConcreteMixed>
// CHECK:     func.call_indirect
// CHECK:     moore.blocking_assign %result
// CHECK:     moore.return
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module testAbstractMixed;
    ConcreteMixed obj;
    int result;
    initial begin
        obj = new;
        result = obj.pureMethod();
    end
endmodule

// Test interface class (pure abstract) - pure virtual methods now have %this argument
// CHECK-LABEL: moore.class.classdecl @PureAbstractInterface {
// CHECK:   moore.class.methoddecl @interfaceMethod -> @"PureAbstractInterface::interfaceMethod" : (!moore.class<@PureAbstractInterface>) -> ()
// CHECK: }

interface class PureAbstractInterface;
    pure virtual function void interfaceMethod();
endclass

// CHECK-LABEL: moore.class.classdecl @ImplementsInterface implements [@PureAbstractInterface] {
// CHECK:   moore.class.methoddecl @interfaceMethod -> @"ImplementsInterface::interfaceMethod" : (!moore.class<@ImplementsInterface>) -> ()
// CHECK: }

class ImplementsInterface implements PureAbstractInterface;
    virtual function void interfaceMethod();
    endfunction
endclass

//===----------------------------------------------------------------------===//
// Combined test: Static via instance with parameterized class
//===----------------------------------------------------------------------===//
// Test the intersection of fixes 1 and 3: accessing static property via
// instance on a parameterized class.

// CHECK-LABEL: moore.class.classdecl @ParamWithStatic {
// CHECK:   moore.class.propertydecl @data : !moore.l8
// CHECK: }
// CHECK-LABEL: moore.class.classdecl @ParamWithStatic_{{[0-9]+}} {
// CHECK:   moore.class.propertydecl @data : !moore.l16
// CHECK: }
// CHECK: moore.global_variable @"ParamWithStatic::counter" : !moore.i32
// CHECK: moore.global_variable @"ParamWithStatic_{{[0-9]+}}::counter" : !moore.i32

class ParamWithStatic #(int SIZE = 8);
    logic [SIZE-1:0] data;
    static int counter;
endclass

// CHECK-LABEL: moore.module @testParamStaticViaInstance() {
// CHECK:   moore.procedure initial {
// Static via instance on ParamWithStatic#(8) - default specialization
// CHECK:     moore.get_global_variable @"ParamWithStatic::counter" : <i32>
// Static via instance on ParamWithStatic#(16) - different specialization
// CHECK:     moore.get_global_variable @"ParamWithStatic_{{[0-9]+}}::counter" : <i32>
// CHECK:     moore.return
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module testParamStaticViaInstance;
    ParamWithStatic #(8) obj8;
    ParamWithStatic #(16) obj16;
    initial begin
        // Access static via instance on different specializations
        obj8.counter = 1;
        obj16.counter = 2;
    end
endmodule

// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

/// Flag tests

// CHECK-LABEL: moore.class.classdecl @plain {
// CHECK: }
class plain;
endclass

// CHECK-LABEL: moore.class.classdecl @abstractOnly {
// CHECK: }
virtual class abstractOnly;
endclass

// CHECK-LABEL: moore.class.classdecl @interfaceTestClass {
// CHECK: }
interface class interfaceTestClass;
endclass

/// Interface tests

// CHECK-LABEL: moore.class.classdecl @interfaceTestClass2 implements [@interfaceTestClass] {
// CHECK: }
class interfaceTestClass2 implements interfaceTestClass;
endclass

// CHECK-LABEL: moore.class.classdecl @interfaceTestClass3 implements [@interfaceTestClass] {
// CHECK: }
interface class interfaceTestClass3 extends interfaceTestClass;
endclass

// CHECK-LABEL: moore.class.classdecl @interfaceTestClass4 implements [@interfaceTestClass3] {
// CHECK: }
class interfaceTestClass4 implements interfaceTestClass3;
endclass

/// Inheritance tests

// CHECK-LABEL: moore.class.classdecl @inheritanceTest {
// CHECK: }
class inheritanceTest;
endclass

// CHECK-LABEL: moore.class.classdecl @inheritanceTest2 extends @inheritanceTest {
// CHECK: }
class inheritanceTest2 extends inheritanceTest;
endclass

// Inheritance + interface tests

// CHECK-LABEL: moore.class.classdecl @D extends @plain {
// CHECK: }
class D extends plain;
endclass

// CHECK-LABEL: moore.class.classdecl @Impl1 implements [@interfaceTestClass] {
// CHECK: }
class Impl1 implements interfaceTestClass;
endclass

// CHECK-LABEL: moore.class.classdecl @Impl2 implements [@interfaceTestClass, @interfaceTestClass3] {
// CHECK: }
class Impl2 implements interfaceTestClass, interfaceTestClass3;
endclass

// CHECK-LABEL: moore.class.classdecl @DI extends @D implements [@interfaceTestClass] {
// CHECK: }
class DI extends D implements interfaceTestClass;
endclass

// CHECK-LABEL: moore.class.classdecl @IMulti implements [@interfaceTestClass, @interfaceTestClass3] {
// CHECK: }
interface class IMulti extends interfaceTestClass, interfaceTestClass3;
endclass

/// Property tests

// CHECK-LABEL: moore.class.classdecl @PropertyCombo {
// CHECK:   moore.class.propertydecl @pubAutoI32 : !moore.i32
// CHECK-NEXT:   moore.class.propertydecl @localAutoI32 : !moore.i32 {member_access = 2 : i32}
// CHECK: }
// Static properties become global variables
// CHECK: moore.global_variable @"PropertyCombo::protStatL18" : !moore.l18
class PropertyCombo;
  // public automatic int
  int pubAutoI32;

  // protected static logic [17:0]
  protected static logic [17:0] protStatL18;

  // local automatic int
  local int localAutoI32;
endclass

// Ensure multiple propertys preserve declaration order
// CHECK-LABEL: moore.class.classdecl @PropertyOrder {
// CHECK:   moore.class.propertydecl @a : !moore.i32
// CHECK-NEXT:   moore.class.propertydecl @b : !moore.i32
// CHECK-NEXT:   moore.class.propertydecl @c : !moore.i32
// CHECK: }
class PropertyOrder;
  int a;
  int b;
  int c;
endclass

// Classes within packages
package testPackage;
   // CHECK-LABEL: moore.class.classdecl @"testPackage::testPackageClass" {
   class testPackageClass;
   // CHECK: }
   endclass
endpackage

// CHECK-LABEL: moore.module @testModule() {
// CHECK: }
// CHECK: moore.class.classdecl @"testModule::testModuleClass" {
// CHECK: }
module testModule #();
   class testModuleClass;
   endclass
endmodule

// CHECK-LABEL: moore.class.classdecl @testClass {
// CHECK: }
// CHECK: moore.class.classdecl @"testClass::testClass" {
// CHECK: }
class testClass;
   class testClass;
   endclass // testClass
endclass

/// Check handle variable

// CHECK-LABEL:  moore.module @testModule2() {
// CHECK-NEXT: [[OBJ:%.+]] = moore.variable : <class<@"testModule2::testModuleClass">>
// CHECK-NEXT:     moore.output
// CHECK-NEXT:   }
// CHECK: moore.class.classdecl @"testModule2::testModuleClass" {
// CHECK: }
module testModule2 #();
    class testModuleClass;
    endclass // testModuleClass2
    testModuleClass t;

endmodule

/// Check calls to new without explicit constructor

// CHECK-LABEL: moore.module @testModule3() {
// CHECK: [[T:%.*]] = moore.variable : <class<@"testModule3::testModuleClass">>
// CHECK: moore.procedure initial {
// CHECK:   [[NEW:%.*]] = moore.class.new : <@"testModule3::testModuleClass">
// CHECK:   moore.blocking_assign [[T]], [[NEW]] : class<@"testModule3::testModuleClass">
// CHECK:   moore.return
// CHECK: }
// CHECK: moore.output

module testModule3;
    class testModuleClass;
    endclass
    testModuleClass t;
    initial begin
        t = new;
    end
endmodule

/// Check property read access

// CHECK-LABEL: moore.module @testModule4() {
// CHECK: [[T:%.*]] = moore.variable : <class<@"testModule4::testModuleClass">>
// CHECK: [[RESULT:%.+]] = moore.variable : <i32>
// CHECK: moore.procedure initial {
// CHECK:    [[NEW:%.*]] = moore.class.new : <@"testModule4::testModuleClass">
// CHECK:    moore.blocking_assign [[T]], [[NEW]] : class<@"testModule4::testModuleClass">
// CHECK:    [[CLASSHANDLE:%.+]] = moore.read [[T]] : <class<@"testModule4::testModuleClass">>
// CHECK:    [[REF:%.+]] = moore.class.property_ref [[CLASSHANDLE]][@a] : <@"testModule4::testModuleClass"> -> <i32>
// CHECK:    [[A:%.+]] = moore.read [[REF]]
// CHECK:    moore.blocking_assign [[RESULT]], [[A]] : i32
// CHECK:    moore.return
// CHECK: }
// CHECK: moore.output
// CHECK: }

// CHECK: moore.class.classdecl @"testModule4::testModuleClass" {
// CHECK-NEXT: moore.class.propertydecl @a : !moore.i32
// CHECK: }

module testModule4;
    class testModuleClass;
       int a;
    endclass
    testModuleClass t;
    int result;
    initial begin
        t = new;
        result = t.a;
    end
endmodule

/// Check property write access

// CHECK-LABEL: moore.module @testModule5() {
// CHECK: [[T:%.*]] = moore.variable : <class<@"testModule5::testModuleClass">>
// CHECK: [[RESULT:%.+]] = moore.variable : <i32>
// CHECK: moore.procedure initial {
// CHECK:    [[NEW:%.*]] = moore.class.new : <@"testModule5::testModuleClass">
// CHECK:    moore.blocking_assign [[T]], [[NEW]] : class<@"testModule5::testModuleClass">
// CHECK:    [[CLASSHANDLE:%.+]] = moore.read [[T]] : <class<@"testModule5::testModuleClass">>
// CHECK:    [[REF:%.+]] = moore.class.property_ref [[CLASSHANDLE]][@a] : <@"testModule5::testModuleClass"> -> <i32>
// CHECK:    [[RESR:%.+]] = moore.read [[RESULT]] : <i32>
// CHECK:    moore.blocking_assign [[REF]], [[RESR]] : i32
// CHECK:    moore.return
// CHECK: }
// CHECK: moore.output
// CHECK: }

// CHECK: moore.class.classdecl @"testModule5::testModuleClass" {
// CHECK-NEXT: moore.class.propertydecl @a : !moore.i32
// CHECK: }

module testModule5;
    class testModuleClass;
       int a;
    endclass
    testModuleClass t;
    int result;
    initial begin
        t = new;
        t.a = result;
    end
endmodule

/// Check implicit upcast

// CHECK-LABEL: moore.module @testModule6() {
// CHECK: [[T:%.*]] = moore.variable : <class<@"testModule6::testModuleClass2">>
// CHECK: [[RESULT:%.+]] = moore.variable : <i32>
// CHECK: moore.procedure initial {
// CHECK:    [[NEW:%.*]] = moore.class.new : <@"testModule6::testModuleClass2">
// CHECK:    moore.blocking_assign [[T]], [[NEW]] : class<@"testModule6::testModuleClass2">
// CHECK:    [[CLASSHANDLE:%.+]] = moore.read [[T]] : <class<@"testModule6::testModuleClass2">>
// CHECK:    [[UPCAST:%.+]] = moore.class.upcast [[CLASSHANDLE]] : <@"testModule6::testModuleClass2"> to <@"testModule6::testModuleClass">
// CHECK:    [[REF:%.+]] = moore.class.property_ref [[UPCAST]][@a] : <@"testModule6::testModuleClass"> -> <i32>
// CHECK:    [[A:%.+]] = moore.read [[REF]]
// CHECK:    moore.blocking_assign [[RESULT]], [[A]] : i32
// CHECK:    moore.return
// CHECK: }
// CHECK: moore.output
// CHECK: }

// CHECK: moore.class.classdecl @"testModule6::testModuleClass" {
// CHECK-NEXT: moore.class.propertydecl @a : !moore.i32
// CHECK: }

module testModule6;
    class testModuleClass;
       int a;
    endclass
    class testModuleClass2 extends testModuleClass;
    endclass
    testModuleClass2 t;
    int result;
    initial begin
        t = new;
        result = t.a;
    end
endmodule

/// Check concrete method calls

// CHECK-LABEL: moore.module @testModule7() {
// CHECK: [[T:%.*]] = moore.variable : <class<@"testModule7::testModuleClass">>
// CHECK: [[RESULT:%.+]] = moore.variable : <i32>
// CHECK: moore.procedure initial {
// CHECK:    [[NEW:%.*]] = moore.class.new : <@"testModule7::testModuleClass">
// CHECK:    moore.blocking_assign [[T]], [[NEW]] : class<@"testModule7::testModuleClass">
// CHECK:    [[CALLREAD:%.+]] = moore.read [[T]] : <class<@"testModule7::testModuleClass">>
// CHECK:    [[FUNCRET:%.+]] = func.call @"testModule7::testModuleClass::returnA"([[CALLREAD]]) : (!moore.class<@"testModule7::testModuleClass">) -> !moore.i32
// CHECK:    moore.blocking_assign [[RESULT]], [[FUNCRET]] : i32
// CHECK:    moore.return
// CHECK: }
// CHECK: moore.output
// CHECK: }

// CHECK: moore.class.classdecl @"testModule7::testModuleClass" {
// CHECK-NEXT: moore.class.propertydecl @a : !moore.i32
// CHECK: }

// CHECK: func.func private @"testModule7::testModuleClass::returnA"
// CHECK-SAME: ([[ARG:%.+]]: !moore.class<@"testModule7::testModuleClass">)
// CHECK-SAME: -> !moore.i32 {
// CHECK-NEXT: [[REF:%.+]] = moore.class.property_ref [[ARG]][@a] : <@"testModule7::testModuleClass"> -> <i32>
// CHECK-NEXT: [[RETURN:%.+]] = moore.read [[REF]] : <i32>
// CHECK-NEXT: return [[RETURN]] : !moore.i32
// CHECK-NEXT: }

module testModule7;
    class testModuleClass;
       int a;
       function int returnA();
          return a;
       endfunction
    endclass
    testModuleClass t;
    int result;
    initial begin
        t = new;
        result = t.returnA();
    end
endmodule


/// Check inherited property access

 // CHECK-LABEL: moore.module @testModule8() {
 // CHECK:    [[t:%.+]] = moore.variable : <class<@"testModule8::testModuleClass2">>
 // CHECK:    [[result:%.+]] = moore.variable : <i32>
 // CHECK:    moore.procedure initial {
 // CHECK:      [[NEW:%.+]] = moore.class.new : <@"testModule8::testModuleClass2">
 // CHECK:      moore.blocking_assign [[t]], [[NEW]] : class<@"testModule8::testModuleClass2">
// CHECK:    [[CALLREAD:%.+]] = moore.read [[t]] : <class<@"testModule8::testModuleClass2">>
// CHECK:       [[CALL:%.+]] = func.call @"testModule8::testModuleClass2::returnA"([[CALLREAD]]) : (!moore.class<@"testModule8::testModuleClass2">) -> !moore.i32
// CHECK:       moore.blocking_assign [[result]], [[CALL]] : i32
 // CHECK:      moore.return
 // CHECK:    }
 // CHECK:    moore.output
 // CHECK:  }
 // CHECK:  moore.class.classdecl @"testModule8::testModuleClass" {
 // CHECK:    moore.class.propertydecl @a : !moore.i32
 // CHECK:  }
 // CHECK:  moore.class.classdecl @"testModule8::testModuleClass2" extends @"testModule8::testModuleClass" {
 // CHECK:  }
 // CHECK:  func.func private @"testModule8::testModuleClass2::returnA"([[ARG:%.+]]: !moore.class<@"testModule8::testModuleClass2">) -> !moore.i32 {
 // CHECK:   [[UPCAST:%.+]] = moore.class.upcast [[ARG]] : <@"testModule8::testModuleClass2"> to <@"testModule8::testModuleClass">
 // CHECK:   [[PROPREF:%.+]] = moore.class.property_ref [[UPCAST]][@a] : <@"testModule8::testModuleClass"> -> <i32>
 // CHECK:   [[RET:%.+]] = moore.read [[PROPREF]] : <i32>
 // CHECK:   return [[RET]] : !moore.i32
 // CHECK: }

module testModule8;

    class testModuleClass;
       int a;
    endclass // testModuleClass

   class testModuleClass2 extends testModuleClass;
       function int returnA();
          return a;
       endfunction
   endclass // testModuleClass2

    testModuleClass2 t;
    int result;
    initial begin
        t = new;
        result = t.returnA();
    end

endmodule

/// Check method lowering without qualified handle

// CHECK-LABEL: moore.module @testModule9() {
// CHECK: [[t:%.+]] = moore.variable : <class<@"testModule9::testModuleClass2">>
// CHECK: [[result:%.+]] = moore.variable : <i32>
// CHECK: moore.procedure initial {
// CHECK:   [[new_obj:%.+]] = moore.class.new : <@"testModule9::testModuleClass2">
// CHECK:   moore.blocking_assign [[t]], [[new_obj]] : class<@"testModule9::testModuleClass2">
// CHECK:    [[CALLREAD:%.+]] = moore.read [[t]] : <class<@"testModule9::testModuleClass2">>
// CHECK:   [[call_ret:%.+]] = func.call @"testModule9::testModuleClass2::returnA"([[CALLREAD]]) : (!moore.class<@"testModule9::testModuleClass2">) -> !moore.i32
// CHECK:   moore.blocking_assign [[result]], [[call_ret]] : i32
// CHECK:   moore.return
// CHECK: }
// CHECK: moore.output
// CHECK: }
// CHECK: moore.class.classdecl @"testModule9::testModuleClass" {
// CHECK:   moore.class.propertydecl @a : !moore.i32
// CHECK: }
// CHECK: func.func private @"testModule9::testModuleClass::myReturn"([[this_ref:%.+]]: !moore.class<@"testModule9::testModuleClass">) -> !moore.i32 {
// CHECK:   [[prop_ref:%.+]] = moore.class.property_ref [[this_ref]][@a] : <@"testModule9::testModuleClass"> -> <i32>
// CHECK:   [[read_val:%.+]] = moore.read [[prop_ref]] : <i32>
// CHECK:   return [[read_val]] : !moore.i32
// CHECK: }
// CHECK: moore.class.classdecl @"testModule9::testModuleClass2" extends @"testModule9::testModuleClass" {
// CHECK: }
// CHECK: func.func private @"testModule9::testModuleClass2::returnA"([[this_ref2:%.+]]: !moore.class<@"testModule9::testModuleClass2">) -> !moore.i32 {
// CHECK:   [[upcast_ref:%.+]] = moore.class.upcast [[this_ref2]] : <@"testModule9::testModuleClass2"> to <@"testModule9::testModuleClass">
// CHECK:   [[call_myReturn:%.+]] = call @"testModule9::testModuleClass::myReturn"([[upcast_ref]]) : (!moore.class<@"testModule9::testModuleClass">) -> !moore.i32
// CHECK:   return [[call_myReturn]] : !moore.i32
// CHECK: }

module testModule9;

    class testModuleClass;
       int a;
       function int myReturn();
          return a;
       endfunction; // myReturn
    endclass // testModuleClass

   class testModuleClass2 extends testModuleClass;
       function int returnA();
          return myReturn();
       endfunction
   endclass // testModuleClass2

    testModuleClass2 t;
    int result;
    initial begin
        t = new;
        result = t.returnA();
    end

endmodule

/// Check forward declarations

// CHECK-LABEL:  moore.class.classdecl @testModuleClass {
// CHECK:  }
// CHECK:  func.func private @"testModuleClass::testFunction"(%arg0: !moore.class<@testModuleClass>, %arg1: !moore.i32) -> !moore.i32 {
// CHECK:    return %arg1 : !moore.i32
// CHECK:  }

class testModuleClass;
    extern function int testFunction(int a);
endclass

function int testModuleClass::testFunction(int a);
    return a;
endfunction

/// Check that calls to new by classes with ctor call the ctor.

// CHECK-LABEL:  moore.module @testModule10() {
// CHECK:    moore.procedure initial {
// CHECK:      [[NEW:%.+]] = moore.class.new : <@"testModule10::testModuleClass">
// CHECK:      [[CONST:%.+]] = moore.constant 3 : i32
// CHECK:      func.call @"testModule10::testModuleClass::new"([[NEW]], [[CONST]]) : (!moore.class<@"testModule10::testModuleClass">, !moore.i32) -> ()
// CHECK:      [[VAR:%.+]] = moore.variable [[NEW]] : <class<@"testModule10::testModuleClass">>
// CHECK:      moore.return
// CHECK:    }
// CHECK:    moore.output
// CHECK:  }
// CHECK:  moore.class.classdecl @"testModule10::testModuleClass" {
// CHECK:    moore.class.propertydecl @a : !moore.i32
// CHECK:  }
// CHECK:  func.func private @"testModule10::testModuleClass::new"(%arg0: !moore.class<@"testModule10::testModuleClass">, %arg1: !moore.i32) {
// CHECK:    [[NEW:%.+]] = moore.variable %arg1 : <i32>
// CHECK:    [[RNEW:%.+]] = moore.read [[NEW]] : <i32>
// CHECK:    moore.blocking_assign [[NEW]], [[RNEW]] : i32
// CHECK:    return
// CHECK:  }

module testModule10;

    class testModuleClass;
       int a;
        function new(int a);
           a = a;
        endfunction
    endclass // testModuleClass

    initial begin
       static testModuleClass t = new(3);
    end

endmodule

/// Check that calls to new by classes with super ctor call the ctor.

// CHECK-LABEL:  moore.class.classdecl @testModuleClass2 {
// CHECK:    moore.class.propertydecl @a : !moore.i32
// CHECK:  }
// CHECK:  func.func private @"testModuleClass2::new"(%arg0: !moore.class<@testModuleClass2>, %arg1: !moore.i32) {
// CHECK:    [[A:%.+]] = moore.class.property_ref %arg0[@a] : <@testModuleClass2> -> <i32>
// CHECK:    moore.blocking_assign [[A]], %arg1 : i32
// CHECK:    return
// CHECK:  }
// CHECK:  moore.class.classdecl @testModuleClass3 extends @testModuleClass2 {
// CHECK:  }
// CHECK:  func.func private @"testModuleClass3::new"(%arg0: !moore.class<@testModuleClass3>, %arg1: !moore.i32) {
// CHECK:    [[UPCAST:%.+]] = moore.class.upcast %arg0 : <@testModuleClass3> to <@testModuleClass2>
// CHECK:    call @"testModuleClass2::new"([[UPCAST]], %arg1) : (!moore.class<@testModuleClass2>, !moore.i32) -> ()
// CHECK:    return
// CHECK:  }

class testModuleClass2;
    int a;
    function new(int a);
        this.a = a;
    endfunction
endclass // testModuleClass

class testModuleClass3 extends testModuleClass2;
    function new(int a);
        super.new(a);
    endfunction
endclass // testModuleClass

/// Check specialized class decl lowering

// CHECK-LABEL:  moore.module @testModuleParametrized() {
// CHECK:    [[T:%.+]] = moore.variable : <class<@"testModuleParametrized::testModuleClass">>
// CHECK:    [[T2:%.+]] = moore.variable : <class<@"testModuleParametrized::testModuleClass">>
// CHECK:    [[T3:%.+]] = moore.variable : <class<@"testModuleParametrized::testModuleClass_1">>
// CHECK:    moore.output
// CHECK:  }
// CHECK:  moore.class.classdecl @"testModuleParametrized::testModuleClass" {
// CHECK:    moore.class.propertydecl @a : !moore.l32
// CHECK:    moore.class.propertydecl @b : !moore.l4
// CHECK:  }
// CHECK:  moore.class.classdecl @"testModuleParametrized::testModuleClass_1" {
// CHECK:    moore.class.propertydecl @a : !moore.l16
// CHECK:    moore.class.propertydecl @b : !moore.l16
// CHECK:  }

module testModuleParametrized;

    class testModuleClass #(
        parameter int WIDTH=32,
        parameter int Other=16
    );
       logic [WIDTH-1:0] a;
       logic [Other-1:0] b;
    endclass // testModuleClass

   testModuleClass#(.WIDTH(32), .Other(4)) t;
   testModuleClass#(.WIDTH(32), .Other(4)) t2;
   testModuleClass#(.WIDTH(16)) t3;
endmodule

/// A test for getting a PR merged that drops elaboration-time constant AST nodes

// CHECK-LABEL:  moore.class.classdecl @testTypedClass extends @testClassType {
// CHECK-NEXT: moore.class.propertydecl @a : !moore.l1
// CHECK:  }

class testClassType #(
    parameter type t = logic
);
   typedef t bool;
endclass

class testTypedClass extends testClassType;
   bool a;
endclass

/// Check that LValues get generated for ClassProperty assignments

// CHECK-LABEL:  moore.class.classdecl @testLValueClass {
// CHECK:    moore.class.propertydecl @a : !moore.i32
// CHECK:  }
// CHECK:  func.func private @"testLValueClass::adder"(%arg0: !moore.class<@testLValueClass>) {
// CHECK:    [[LVAL:%.+]] = moore.class.property_ref %arg0[@a] : <@testLValueClass> -> <i32>
// CHECK:    [[RLVAL:%.+]] = moore.class.property_ref %arg0[@a] : <@testLValueClass> -> <i32>
// CHECK:    [[RVAL:%.+]] = moore.read [[RLVAL]] : <i32>
// CHECK:    [[CONST:%.+]] = moore.constant 1 : i32
// CHECK:    [[NEWVAL:%.+]] = moore.add [[RVAL]], [[CONST]] : i32
// CHECK:    moore.blocking_assign [[LVAL]], [[NEWVAL]] : i32
// CHECK:    return
// CHECK:  }

class testLValueClass;
int a;
function void adder;
    a = a + 1;
endfunction
endclass

/// Check that inheritance is enforced over specialized classes

// CHECK-LABEL:  moore.class.classdecl @GenericBar {
// CHECK:  }
// CHECK:  moore.class.classdecl @SpecializedFoo extends @GenericBar {
// CHECK:  }

class GenericBar #(int X=0, int Y=1, int Z=2); endclass
localparam x=3, y=4, z=5;

class SpecializedFoo extends GenericBar #(x,y,z); endclass

/// Check virtual attribute of methoddecl

// CHECK-LABEL: moore.class.classdecl @testClassVirtual {
// CHECK-NEXT:    moore.class.methoddecl @testFun -> @"testClassVirtual::testFun" : (!moore.class<@testClassVirtual>) -> ()
// CHECK:  }
// CHECK:  func.func private @"testClassVirtual::testFun"(%arg0: !moore.class<@testClassVirtual>) {
// CHECK:    return
// CHECK:  }

class testClassVirtual;
   virtual function void testFun();
   endfunction
endclass

/// Check virtual dispatch

// CHECK-LABEL: func.func private @testVirtualDispatch
// CHECK-SAME: (%arg0: !moore.class<@testClassVirtual>) {
// CHECK-NEXT:    [[VMETH:%.+]] = moore.vtable.load_method %arg0 : @testFun of <@testClassVirtual> -> (!moore.class<@testClassVirtual>) -> ()
// CHECK-NEXT:    call_indirect [[VMETH]](%arg0) : (!moore.class<@testClassVirtual>) -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }

function void testVirtualDispatch (testClassVirtual t);
    t.testFun();
endfunction

/// Check pure virtual forward declarations

// CHECK-LABEL:  moore.class.classdecl @virtualFunctionClass {
// CHECK:    moore.class.methoddecl @subroutine -> @"virtualFunctionClass::subroutine" : (!moore.class<@virtualFunctionClass>) -> ()
// CHECK:  }
// CHECK:  func.func private @"virtualFunctionClass::subroutine"(%arg0: !moore.class<@virtualFunctionClass>) {
// CHECK:    return
// CHECK:  }
// CHECK:  moore.class.classdecl @realFunctionClass implements [@virtualFunctionClass] {
// CHECK:    moore.class.methoddecl @subroutine -> @"realFunctionClass::subroutine" : (!moore.class<@realFunctionClass>) -> ()
// CHECK:  }
// CHECK:  func.func private @"realFunctionClass::subroutine"(%arg0: !moore.class<@realFunctionClass>) {
// CHECK:    return
// CHECK:  }

interface class virtualFunctionClass;
pure virtual function void subroutine;
endclass

class realFunctionClass implements virtualFunctionClass;
virtual function void subroutine; endfunction
endclass

/// Check randomization support - rand and randc properties

// CHECK-LABEL: moore.class.classdecl @RandomizableClass {
// CHECK-NEXT:    moore.class.propertydecl @data : !moore.i32 rand_mode rand
// CHECK-NEXT:    moore.class.propertydecl @mode : !moore.i8 rand_mode randc
// CHECK-NEXT:    moore.class.propertydecl @fixed : !moore.i16
// CHECK: }

class RandomizableClass;
    rand int data;
    randc byte mode;
    shortint fixed;
endclass

/// Check constraint block support with expression lowering
/// Note: Property references in constraints are not yet fully resolved to
/// `this` accesses, so they appear as comparisons with constant 0.

// CHECK-LABEL: moore.class.classdecl @ConstrainedClass {
// CHECK:         moore.class.propertydecl @x : !moore.i32 rand_mode rand
// CHECK:         moore.constraint.block @valid_range {
// CHECK:           %[[GT:.*]] = moore.sgt {{.*}} : i32 -> i1
// CHECK:           moore.constraint.expr %[[GT]] : i1
// CHECK:           %[[LT:.*]] = moore.slt {{.*}} : i32 -> i1
// CHECK:           moore.constraint.expr %[[LT]] : i1
// CHECK:         }
// CHECK: }

class ConstrainedClass;
    rand int x;
    constraint valid_range { x > 0; x < 100; }
endclass

/// Check static constraint blocks

// CHECK-LABEL: moore.class.classdecl @StaticConstraintClass {
// CHECK:         moore.class.propertydecl @y : !moore.i32 rand_mode rand
// CHECK:         moore.constraint.block static @static_bound {
// CHECK:           %[[GE:.*]] = moore.sge {{.*}} : i32 -> i1
// CHECK:           moore.constraint.expr %[[GE]] : i1
// CHECK:         }
// CHECK: }

class StaticConstraintClass;
    rand int y;
    static constraint static_bound { y >= 0; }
endclass

/// Check implicit constraint block (no body)
// These are forward declarations for constraints that can be defined externally.

// CHECK-LABEL: moore.class.classdecl @ImplicitConstraintClass {
// CHECK:         moore.class.propertydecl @z : !moore.i32 rand_mode rand
// CHECK:         moore.constraint.block @implicit_c {
// CHECK-NOT:       moore.constraint.expr
// CHECK:         }
// CHECK: }

class ImplicitConstraintClass;
    rand int z;
    constraint implicit_c;
endclass

/// Check constraint with if-else and soft constraints

// CHECK-LABEL: moore.class.classdecl @AdvancedConstraints {
// CHECK:         moore.class.propertydecl @x : !moore.i32 rand_mode rand
// CHECK:         moore.class.propertydecl @mode : !moore.i1 rand_mode rand
// CHECK:         moore.constraint.block @soft_c {
// CHECK:           %[[LT:.*]] = moore.slt {{.*}} : i32 -> i1
// CHECK:           moore.constraint.expr %[[LT]] : i1 soft
// CHECK:         }
// CHECK:         moore.constraint.block @impl_c {
// CHECK:           moore.constraint.implication {{.*}} : i1 {
// CHECK:             %[[GT:.*]] = moore.sgt {{.*}} : i32 -> i1
// CHECK:             moore.constraint.expr %[[GT]] : i1
// CHECK:           }
// CHECK:         }
// CHECK:         moore.constraint.block @ifelse_c {
// CHECK:           moore.constraint.if_else {{.*}} : i1 {
// CHECK:             %[[GT2:.*]] = moore.sgt {{.*}} : i32 -> i1
// CHECK:             moore.constraint.expr %[[GT2]] : i1
// CHECK:           } else {
// CHECK:             %[[LE:.*]] = moore.sle {{.*}} : i32 -> i1
// CHECK:             moore.constraint.expr %[[LE]] : i1
// CHECK:           }
// CHECK:         }
// CHECK: }

class AdvancedConstraints;
    rand int x;
    rand bit mode;
    constraint soft_c { soft x < 100; }
    constraint impl_c { mode -> x > 50; }
    constraint ifelse_c {
        if (mode) {
            x > 0;
        } else {
            x <= 0;
        }
    }
endclass

//===----------------------------------------------------------------------===//
// Interface Tests
//===----------------------------------------------------------------------===//

/// Check basic interface declaration with signals
/// Note: Interfaces are only emitted when instantiated or used by modules,
/// not when only referenced by virtual interfaces in classes.

interface basic_bus;
    logic clk;
    logic [31:0] data;
    logic valid;
endinterface

/// Check interface with modports

interface handshake_bus;
    logic clk;
    logic [7:0] data;
    logic valid;
    logic ready;

    modport driver (output clk, output data, output valid, input ready);
    modport receiver (input clk, input data, input valid, output ready);
endinterface

/// Check virtual interface variable type

// CHECK-LABEL: moore.class.classdecl @VifDriver {
// CHECK-NEXT:    moore.class.propertydecl @vif : !moore.virtual_interface<@handshake_bus::@driver>
// CHECK: }

class VifDriver;
    virtual handshake_bus.driver vif;
endclass

/// Check virtual interface without modport

// CHECK-LABEL: moore.class.classdecl @VifHolder {
// CHECK-NEXT:    moore.class.propertydecl @bus : !moore.virtual_interface<@basic_bus>
// CHECK: }

class VifHolder;
    virtual basic_bus bus;
endclass

/// Check interface with ports (like SPI interface)

// CHECK-LABEL: moore.interface @SpiInterface {
// CHECK-NEXT:    moore.interface.signal @pclk : !moore.l1
// CHECK-NEXT:    moore.interface.signal @areset : !moore.l1
// CHECK-NEXT:    moore.interface.signal @mosi : !moore.l1
// CHECK-NEXT:    moore.interface.signal @miso : !moore.l1
// CHECK-NEXT:    moore.interface.signal @sclk : !moore.l1
// CHECK-NEXT:    moore.interface.modport @master (output @mosi, input @miso, output @sclk)
// CHECK-NEXT:    moore.interface.modport @slave (input @mosi, output @miso, input @sclk)
// CHECK: }

interface SpiInterface(input pclk, input areset);
    logic mosi;
    logic miso;
    logic sclk;
    modport master(output mosi, input miso, output sclk);
    modport slave(input mosi, output miso, input sclk);
endinterface

/// Check interface instantiation inside a module

// CHECK-LABEL: moore.module @interface_inst_test
// CHECK:         %spi = moore.interface.instance @SpiInterface : <virtual_interface<@SpiInterface>>

module interface_inst_test(input clk);
    SpiInterface spi(clk, 1'b0);
endmodule

/// Check $cast dynamic type checking for class downcasts

// CHECK-LABEL: moore.class.classdecl @BaseCastClass {
// CHECK: }

class BaseCastClass;
endclass

// CHECK-LABEL: moore.class.classdecl @DerivedCastClass extends @BaseCastClass {
// CHECK: }

class DerivedCastClass extends BaseCastClass;
endclass

// CHECK-LABEL: moore.module @testCastModule() {
// CHECK:   [[DERIVEDVAR:%.+]] = moore.variable : <class<@DerivedCastClass>>
// CHECK:   [[BASEVAR:%.+]] = moore.variable : <class<@BaseCastClass>>
// CHECK:   [[RESULT:%.+]] = moore.variable : <i32>
// CHECK:   moore.procedure initial {
// CHECK:     [[NEW:%.+]] = moore.class.new : <@DerivedCastClass>
// CHECK:     moore.blocking_assign [[DERIVEDVAR]], [[NEW]] : class<@DerivedCastClass>
// CHECK:     [[DERIVEDVAL:%.+]] = moore.read [[DERIVEDVAR]] : <class<@DerivedCastClass>>
// CHECK:     [[UPCAST:%.+]] = moore.class.upcast [[DERIVEDVAL]] : <@DerivedCastClass> to <@BaseCastClass>
// CHECK:     moore.blocking_assign [[BASEVAR]], [[UPCAST]] : class<@BaseCastClass>
// CHECK:     [[BASEVAL:%.+]] = moore.read [[BASEVAR]] : <class<@BaseCastClass>>
// CHECK:     [[DYNCAST:%.+]], [[SUCCESS:%.+]] = moore.class.dyn_cast [[BASEVAL]] : <@BaseCastClass> to <@DerivedCastClass>
// CHECK:     moore.blocking_assign [[DERIVEDVAR]], [[DYNCAST]] : class<@DerivedCastClass>
// CHECK:     [[SUCCESSINT:%.+]] = moore.conversion [[SUCCESS]] : i1 -> !moore.i32
// CHECK:     moore.blocking_assign [[RESULT]], [[SUCCESSINT]] : i32
// CHECK:     moore.return
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module testCastModule;
    DerivedCastClass derived;
    BaseCastClass base;
    int result;
    initial begin
        derived = new;
        base = derived;
        result = $cast(derived, base);
    end
endmodule

/// Check $cast with sibling classes (expected to fail at runtime)
/// This tests the RTTI infrastructure - casting between unrelated types should fail

// CHECK-LABEL: moore.class.classdecl @SiblingA extends @BaseCastClass {
// CHECK: }

class SiblingA extends BaseCastClass;
endclass

// CHECK-LABEL: moore.class.classdecl @SiblingB extends @BaseCastClass {
// CHECK: }

class SiblingB extends BaseCastClass;
endclass

// CHECK-LABEL: moore.module @testCastSiblingClasses() {
// CHECK:   [[SIBLING_A:%.+]] = moore.variable : <class<@SiblingA>>
// CHECK:   [[SIBLING_B:%.+]] = moore.variable : <class<@SiblingB>>
// CHECK:   [[BASE:%.+]] = moore.variable : <class<@BaseCastClass>>
// CHECK:   [[RESULT:%.+]] = moore.variable : <i32>
// CHECK:   moore.procedure initial {
// CHECK:     [[NEW_A:%.+]] = moore.class.new : <@SiblingA>
// CHECK:     moore.blocking_assign [[SIBLING_A]], [[NEW_A]] : class<@SiblingA>
// CHECK:     [[VAL_A:%.+]] = moore.read [[SIBLING_A]] : <class<@SiblingA>>
// CHECK:     [[UPCAST_A:%.+]] = moore.class.upcast [[VAL_A]] : <@SiblingA> to <@BaseCastClass>
// CHECK:     moore.blocking_assign [[BASE]], [[UPCAST_A]] : class<@BaseCastClass>
// CHECK:     [[BASE_VAL:%.+]] = moore.read [[BASE]] : <class<@BaseCastClass>>
// CHECK:     [[DYNCAST:%.+]], [[SUCCESS:%.+]] = moore.class.dyn_cast [[BASE_VAL]] : <@BaseCastClass> to <@SiblingB>
// CHECK:     moore.blocking_assign [[SIBLING_B]], [[DYNCAST]] : class<@SiblingB>
// CHECK:     [[SUCCESSINT:%.+]] = moore.conversion [[SUCCESS]] : i1 -> !moore.i32
// CHECK:     moore.blocking_assign [[RESULT]], [[SUCCESSINT]] : i32
// CHECK:     moore.return
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module testCastSiblingClasses;
    SiblingA siblingA;
    SiblingB siblingB;
    BaseCastClass base;
    int result;
    initial begin
        // Create a SiblingA object and upcast to base
        siblingA = new;
        base = siblingA;
        // Try to cast to SiblingB - this should return 0 (fail) at runtime
        // because SiblingA is not a SiblingB
        result = $cast(siblingB, base);
        // At runtime, result should be 0 (cast failed)
    end
endmodule

/// Check class handle comparison with null

// CHECK-LABEL: moore.class.classdecl @CmpTestClass {
// CHECK: }

class CmpTestClass;
endclass

// CHECK-LABEL: moore.class.classdecl @CmpDerivedClass extends @CmpTestClass {
// CHECK: }

class CmpDerivedClass extends CmpTestClass;
endclass

// CHECK-LABEL: moore.module @testHandleCmpNull
// CHECK: [[T:%.+]] = moore.variable : <class<@CmpTestClass>>
// CHECK: [[RESULT:%.+]] = moore.variable : <i1>
// CHECK: moore.procedure initial {
// CHECK:   [[VAL:%.+]] = moore.read [[T]] : <class<@CmpTestClass>>
// CHECK:   [[NULL:%.+]] = moore.class.null : <@CmpTestClass>
// CHECK:   [[CMP:%.+]] = moore.class_handle_cmp eq [[VAL]], [[NULL]] : <@CmpTestClass>
// CHECK:   moore.blocking_assign [[RESULT]], [[CMP]] : i1
// CHECK:   moore.return
// CHECK: }

module testHandleCmpNull;
    CmpTestClass t;
    bit result;
    initial begin
        result = (t == null);
    end
endmodule

/// Check class handle inequality comparison with null

// CHECK-LABEL: moore.module @testHandleCmpNullNe
// CHECK: [[T:%.+]] = moore.variable : <class<@CmpTestClass>>
// CHECK: [[RESULT:%.+]] = moore.variable : <i1>
// CHECK: moore.procedure initial {
// CHECK:   [[VAL:%.+]] = moore.read [[T]] : <class<@CmpTestClass>>
// CHECK:   [[NULL:%.+]] = moore.class.null : <@CmpTestClass>
// CHECK:   [[CMP:%.+]] = moore.class_handle_cmp ne [[VAL]], [[NULL]] : <@CmpTestClass>
// CHECK:   moore.blocking_assign [[RESULT]], [[CMP]] : i1
// CHECK:   moore.return
// CHECK: }

module testHandleCmpNullNe;
    CmpTestClass t;
    bit result;
    initial begin
        result = (t != null);
    end
endmodule

/// Check handle-to-handle comparison (same type)

// CHECK-LABEL: moore.module @testHandleCmpHandles
// CHECK: [[T1:%.+]] = moore.variable : <class<@CmpTestClass>>
// CHECK: [[T2:%.+]] = moore.variable : <class<@CmpTestClass>>
// CHECK: [[RESULT:%.+]] = moore.variable : <i1>
// CHECK: moore.procedure initial {
// CHECK:   [[V1:%.+]] = moore.read [[T1]] : <class<@CmpTestClass>>
// CHECK:   [[V2:%.+]] = moore.read [[T2]] : <class<@CmpTestClass>>
// CHECK:   [[CMP:%.+]] = moore.class_handle_cmp eq [[V1]], [[V2]] : <@CmpTestClass>
// CHECK:   moore.blocking_assign [[RESULT]], [[CMP]] : i1
// CHECK:   moore.return
// CHECK: }

module testHandleCmpHandles;
    CmpTestClass t1, t2;
    bit result;
    initial begin
        result = (t1 == t2);
    end
endmodule

/// Check comparison with derived class (upcasting)

// CHECK-LABEL: moore.module @testHandleCmpDerived
// CHECK: [[BASE:%.+]] = moore.variable : <class<@CmpTestClass>>
// CHECK: [[DERIVED:%.+]] = moore.variable : <class<@CmpDerivedClass>>
// CHECK: [[RESULT:%.+]] = moore.variable : <i1>
// CHECK: moore.procedure initial {
// CHECK:   [[VBASE:%.+]] = moore.read [[BASE]] : <class<@CmpTestClass>>
// CHECK:   [[VDERIVED:%.+]] = moore.read [[DERIVED]] : <class<@CmpDerivedClass>>
// CHECK:   [[UPCAST:%.+]] = moore.class.upcast [[VDERIVED]] : <@CmpDerivedClass> to <@CmpTestClass>
// CHECK:   [[CMP:%.+]] = moore.class_handle_cmp eq [[VBASE]], [[UPCAST]] : <@CmpTestClass>
// CHECK:   moore.blocking_assign [[RESULT]], [[CMP]] : i1
// CHECK:   moore.return
// CHECK: }

module testHandleCmpDerived;
    CmpTestClass base;
    CmpDerivedClass derived;
    bit result;
    initial begin
        result = (base == derived);
    end
endmodule

//===----------------------------------------------------------------------===//
// Static Class Property Tests
//===----------------------------------------------------------------------===//

/// Check static class property declaration becomes a global variable

// CHECK-LABEL: moore.class.classdecl @StaticPropertyClass {
// CHECK: }
// CHECK: moore.global_variable @"StaticPropertyClass::counter" : !moore.i32

class StaticPropertyClass;
    static int counter;
endclass

/// Check static class property access in static function

// CHECK-LABEL: moore.class.classdecl @StaticMethodClass {
// CHECK: }
// CHECK: moore.global_variable @"StaticMethodClass::m_inst" : !moore.class<@StaticMethodClass>

class StaticMethodClass;
    static local StaticMethodClass m_inst;
    static function StaticMethodClass get();
        if (m_inst == null) m_inst = new();
        return m_inst;
    endfunction
endclass

/// Check static class property read from module

// CHECK-LABEL: moore.class.classdecl @StaticReadClass {
// CHECK: }
// CHECK: moore.global_variable @"StaticReadClass::value" : !moore.i32

class StaticReadClass;
    static int value;
endclass

// CHECK-LABEL: moore.module @testStaticRead() {
// CHECK:   %result = moore.variable : <i32>
// CHECK:   moore.procedure initial {
// CHECK:     %[[GLOBAL:.+]] = moore.get_global_variable @"StaticReadClass::value" : <i32>
// CHECK:     %[[VAL:.+]] = moore.read %[[GLOBAL]] : <i32>
// CHECK:     moore.blocking_assign %result, %[[VAL]] : i32
// CHECK:     moore.return
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module testStaticRead;
    int result;
    initial begin
        result = StaticReadClass::value;
    end
endmodule

/// Check static class property write from module

// CHECK-LABEL: moore.module @testStaticWrite() {
// CHECK:   moore.procedure initial {
// CHECK:     %[[GLOBAL:.+]] = moore.get_global_variable @"StaticReadClass::value" : <i32>
// CHECK:     %[[CONST:.+]] = moore.constant 42 : i32
// CHECK:     moore.blocking_assign %[[GLOBAL]], %[[CONST]] : i32
// CHECK:     moore.return
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module testStaticWrite;
    initial begin
        StaticReadClass::value = 42;
    end
endmodule

/// Check null literal and class handle comparison

// CHECK-LABEL: moore.class.classdecl @NullTestClass {
// CHECK:   moore.class.propertydecl @data : !moore.i32
// CHECK: }

class NullTestClass;
    int data;
endclass

// CHECK-LABEL: moore.module @testNullComparison() {
// CHECK:   [[OBJ:%.+]] = moore.variable : <class<@NullTestClass>>
// CHECK:   [[FLAG:%.+]] = moore.variable : <i32>
// CHECK:   moore.procedure initial {
// CHECK:     [[NEW:%.+]] = moore.class.new : <@NullTestClass>
// CHECK:     moore.blocking_assign [[OBJ]], [[NEW]] : class<@NullTestClass>
// Test obj == null: should generate ClassNullOp and ClassHandleCmpOp
// CHECK:     [[OBJV1:%.+]] = moore.read [[OBJ]] : <class<@NullTestClass>>
// The null literal is first created with __null__ type, then properly typed for comparison
// CHECK:     moore.class.null : <@__null__>
// CHECK:     [[NULL1:%.+]] = moore.class.null : <@NullTestClass>
// CHECK:     moore.class_handle_cmp eq [[OBJV1]], [[NULL1]] : <@NullTestClass> -> i1
// Test obj != null: should generate ClassNullOp and ClassHandleCmpOp with ne predicate
// CHECK:     [[OBJV2:%.+]] = moore.read [[OBJ]] : <class<@NullTestClass>>
// CHECK:     moore.class.null : <@__null__>
// CHECK:     [[NULL2:%.+]] = moore.class.null : <@NullTestClass>
// CHECK:     moore.class_handle_cmp ne [[OBJV2]], [[NULL2]] : <@NullTestClass> -> i1
// CHECK:     moore.return
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module testNullComparison;
    NullTestClass obj;
    int flag;
    initial begin
        obj = new;
        // Test comparison with null using ==
        if (obj == null) begin
            flag = 0;
        end
        // Test comparison with null using !=
        if (obj != null) begin
            flag = 1;
        end
    end
endmodule

//===----------------------------------------------------------------------===//
// Built-in Class Tests (semaphore, mailbox)
//===----------------------------------------------------------------------===//

/// Check semaphore new() construction

// CHECK-LABEL: moore.module @testSemaphoreNew() {
// CHECK:   %sem = moore.variable : <class<@"std::semaphore">>
// CHECK:   moore.procedure initial {
// Default new (no args)
// CHECK:     %{{[0-9]+}} = moore.class.new : <@"std::semaphore">
// CHECK:     moore.blocking_assign %sem, %{{[0-9]+}} : class<@"std::semaphore">
// new with keyCount argument
// CHECK:     %{{[0-9]+}} = moore.class.new : <@"std::semaphore">
// CHECK:     moore.blocking_assign %sem, %{{[0-9]+}} : class<@"std::semaphore">
// CHECK:     moore.return
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module testSemaphoreNew;
    semaphore sem;
    initial begin
        // Test default constructor (no args)
        sem = new;
        // Test constructor with initial key count
        sem = new(1);
    end
endmodule

/// Check mailbox new() construction

// CHECK-LABEL: moore.module @testMailboxNew() {
// CHECK:   %typedMb = moore.variable : <class<@"std::mailbox">>
// CHECK:   %untypedMb = moore.variable : <class<@"std::mailbox_0">>
// CHECK:   moore.procedure initial {
// Default new for typed mailbox
// CHECK:     %{{[0-9]+}} = moore.class.new : <@"std::mailbox">
// CHECK:     moore.blocking_assign %typedMb, %{{[0-9]+}} : class<@"std::mailbox">
// new with bound argument for typed mailbox
// CHECK:     %{{[0-9]+}} = moore.class.new : <@"std::mailbox">
// CHECK:     moore.blocking_assign %typedMb, %{{[0-9]+}} : class<@"std::mailbox">
// Default new for untyped mailbox
// CHECK:     %{{[0-9]+}} = moore.class.new : <@"std::mailbox_0">
// CHECK:     moore.blocking_assign %untypedMb, %{{[0-9]+}} : class<@"std::mailbox_0">
// new with bound argument for untyped mailbox
// CHECK:     %{{[0-9]+}} = moore.class.new : <@"std::mailbox_0">
// CHECK:     moore.blocking_assign %untypedMb, %{{[0-9]+}} : class<@"std::mailbox_0">
// CHECK:     moore.return
// CHECK:   }
// CHECK:   moore.output
// CHECK: }

module testMailboxNew;
    mailbox #(int) typedMb;
    mailbox untypedMb;
    initial begin
        // Test default constructor for typed mailbox
        typedMb = new;
        // Test constructor with bound for typed mailbox
        typedMb = new(10);
        // Test default constructor for untyped mailbox
        untypedMb = new;
        // Test constructor with bound for untyped mailbox
        untypedMb = new(5);
    end
endmodule

/// Check semaphore in class (UVM pattern)

// CHECK-LABEL: moore.class.classdecl @UvmSequenceBase {
// CHECK:   moore.class.propertydecl @m_mutex : !moore.class<@"std::semaphore">
// CHECK: }
// CHECK: func.func private @"UvmSequenceBase::new"(%arg0: !moore.class<@UvmSequenceBase>) {
// CHECK:   %{{[0-9]+}} = moore.class.property_ref %arg0[@m_mutex] : <@UvmSequenceBase> -> <class<@"std::semaphore">>
// CHECK:   %{{[0-9]+}} = moore.class.new : <@"std::semaphore">
// CHECK:   moore.blocking_assign %{{[0-9]+}}, %{{[0-9]+}} : class<@"std::semaphore">
// CHECK:   return
// CHECK: }

class UvmSequenceBase;
    semaphore m_mutex;
    function new();
        m_mutex = new(1);
    endfunction
endclass

/// Check interface class upcast (IEEE 1800-2017 Section 8.26.5)
/// A class that implements an interface class can be assigned to a variable
/// of that interface class type.

// CHECK-LABEL: moore.class.classdecl @ihello {
// CHECK: }
interface class ihello;
    pure virtual function void hello();
endclass

// CHECK-LABEL: moore.class.classdecl @Hello implements [@ihello] {
// CHECK:   moore.class.methoddecl @hello -> @"Hello::hello"
// CHECK: }
class Hello implements ihello;
    virtual function void hello();
        $display("hello world");
    endfunction
endclass

// CHECK: func.func private @"Hello::hello"

// CHECK-LABEL: moore.module @testInterfaceClassUpcast() {
// CHECK: [[OBJ:%.*]] = moore.variable : <class<@Hello>>
// CHECK: [[IOBJ:%.*]] = moore.variable : <class<@ihello>>
// CHECK: moore.procedure initial {
// CHECK:    [[NEW:%.*]] = moore.class.new : <@Hello>
// CHECK:    moore.blocking_assign [[OBJ]], [[NEW]] : class<@Hello>
// CHECK:    [[OBJREAD:%.+]] = moore.read [[OBJ]] : <class<@Hello>>
// CHECK:    [[UPCAST:%.+]] = moore.class.upcast [[OBJREAD]] : <@Hello> to <@ihello>
// CHECK:    moore.blocking_assign [[IOBJ]], [[UPCAST]] : class<@ihello>
// CHECK:    moore.return
// CHECK: }
// CHECK: moore.output
// CHECK: }
module testInterfaceClassUpcast;
    Hello obj;
    ihello iobj;
    initial begin
        obj = new;
        iobj = obj;
    end
endmodule

/// Check shallow copy (IEEE 1800-2017 Section 8.12)
/// The `new <source>` syntax creates a shallow copy of a class instance.

// CHECK-LABEL: moore.class.classdecl @CopyableClass {
// CHECK:   moore.class.propertydecl @value : !moore.i32
// CHECK: }
class CopyableClass;
    int value;
    task set_value(int v);
        value = v;
    endtask
endclass

// CHECK-LABEL: moore.module @testShallowCopy() {
// CHECK: [[OBJ0:%.*]] = moore.variable : <class<@CopyableClass>>
// CHECK: [[OBJ1:%.*]] = moore.variable : <class<@CopyableClass>>
// CHECK: moore.procedure initial {
// CHECK:    [[NEW:%.*]] = moore.class.new : <@CopyableClass>
// CHECK:    moore.blocking_assign [[OBJ0]], [[NEW]] : class<@CopyableClass>
// CHECK:    [[OBJ0_READ1:%.+]] = moore.read [[OBJ0]] : <class<@CopyableClass>>
// CHECK:    [[VALUE_REF:%.+]] = moore.class.property_ref [[OBJ0_READ1]][@value]
// CHECK:    [[CONST42:%.+]] = moore.constant 42 : i32
// CHECK:    moore.blocking_assign [[VALUE_REF]], [[CONST42]] : i32
// CHECK:    [[OBJ0_READ2:%.+]] = moore.read [[OBJ0]] : <class<@CopyableClass>>
// CHECK:    [[COPY:%.*]] = moore.class.copy [[OBJ0_READ2]] : <@CopyableClass>
// CHECK:    moore.blocking_assign [[OBJ1]], [[COPY]] : class<@CopyableClass>
// CHECK:    moore.return
// CHECK: }
// CHECK: moore.output
// CHECK: }
module testShallowCopy;
    CopyableClass obj0;
    CopyableClass obj1;
    initial begin
        obj0 = new;
        obj0.value = 42;
        obj1 = new obj0;
    end
endmodule

/// Check class parameter access (IEEE 1800-2017 Section 8.25)
/// Class parameters can be accessed like properties but are compile-time constants.

// CHECK-LABEL: moore.class.classdecl @ParameterizedClass {
// CHECK: }
class ParameterizedClass #(parameter int VALUE = 10);
endclass

// CHECK-LABEL: moore.module @testClassParameter() {
// CHECK: [[OBJ:%.*]] = moore.variable : <class<@ParameterizedClass>>
// CHECK: [[RESULT:%.*]] = moore.variable : <i32>
// CHECK: moore.procedure initial {
// CHECK:    [[NEW:%.*]] = moore.class.new : <@ParameterizedClass>
// CHECK:    moore.blocking_assign [[OBJ]], [[NEW]] : class<@ParameterizedClass>
// CHECK:    [[CONST:%.*]] = moore.constant 42 : i32
// CHECK:    moore.blocking_assign [[RESULT]], [[CONST]] : i32
// CHECK:    moore.return
// CHECK: }
// CHECK: moore.output
// CHECK: }
module testClassParameter;
    ParameterizedClass #(42) obj;
    int result;
    initial begin
        obj = new;
        result = obj.VALUE;
    end
endmodule

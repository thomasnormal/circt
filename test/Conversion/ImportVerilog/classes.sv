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
// CHECK-NEXT:   moore.class.propertydecl @protStatL18 : !moore.l18
// CHECK-NEXT:   moore.class.propertydecl @localAutoI32 : !moore.i32
// CHECK: }
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
// CHECK:    [[T3:%.+]] = moore.variable : <class<@"testModuleParametrized::testModuleClass_0">>
// CHECK:    moore.output
// CHECK:  }
// CHECK:  moore.class.classdecl @"testModuleParametrized::testModuleClass" {
// CHECK:    moore.class.propertydecl @a : !moore.l32
// CHECK:    moore.class.propertydecl @b : !moore.l4
// CHECK:  }
// CHECK:  moore.class.classdecl @"testModuleParametrized::testModuleClass_0" {
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
// CHECK:    moore.class.methoddecl @subroutine : (!moore.class<@virtualFunctionClass>)
// CHECK:  }
// CHECK:  moore.class.classdecl @realFunctionClass implements [@virtualFunctionClass] {
// CHECK:    moore.class.methoddecl @subroutine -> @"realFunctionClass::subroutine" : (!moore.class<@realFunctionClass>)
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

/// Check constraint block support

// CHECK-LABEL: moore.class.classdecl @ConstrainedClass {
// CHECK-NEXT:    moore.class.propertydecl @x : !moore.i32 rand_mode rand
// CHECK-NEXT:    moore.constraint.block @valid_range {
// CHECK-NEXT:    }
// CHECK: }

class ConstrainedClass;
    rand int x;
    constraint valid_range { x > 0; x < 100; }
endclass

/// Check static constraint blocks

// CHECK-LABEL: moore.class.classdecl @StaticConstraintClass {
// CHECK-NEXT:    moore.class.propertydecl @y : !moore.i32 rand_mode rand
// CHECK-NEXT:    moore.constraint.block static @static_bound {
// CHECK-NEXT:    }
// CHECK: }

class StaticConstraintClass;
    rand int y;
    static constraint static_bound { y >= 0; }
endclass

//===----------------------------------------------------------------------===//
// Interface Tests
//===----------------------------------------------------------------------===//

/// Check basic interface declaration with signals

// CHECK-LABEL: moore.interface @basic_bus {
// CHECK-NEXT:    moore.interface.signal @clk : !moore.l1
// CHECK-NEXT:    moore.interface.signal @data : !moore.l32
// CHECK-NEXT:    moore.interface.signal @valid : !moore.l1
// CHECK: }

interface basic_bus;
    logic clk;
    logic [31:0] data;
    logic valid;
endinterface

/// Check interface with modports

// CHECK-LABEL: moore.interface @handshake_bus {
// CHECK-NEXT:    moore.interface.signal @clk : !moore.l1
// CHECK-NEXT:    moore.interface.signal @data : !moore.l8
// CHECK-NEXT:    moore.interface.signal @valid : !moore.l1
// CHECK-NEXT:    moore.interface.signal @ready : !moore.l1
// CHECK-NEXT:    moore.interface.modport @driver (output @clk, output @data, output @valid, input @ready)
// CHECK-NEXT:    moore.interface.modport @receiver (input @clk, input @data, input @valid, output @ready)
// CHECK: }

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

/// Check static property access from static methods
/// This tests the fix for static properties accessed from static methods.
/// The fix ensures that when there's no implicit 'this' reference (because
/// we're in a static context), we correctly treat the property as static.

// CHECK-LABEL: moore.class.classdecl @StaticPropertyAccess {
// CHECK-NEXT:    moore.class.propertydecl @count : !moore.i32
// CHECK: }
// CHECK: func.func private @"StaticPropertyAccess::get_count"() -> !moore.i32 {
// CHECK:   %count = moore.variable : <i32>
// CHECK:   [[READ:%.+]] = moore.read %count : <i32>
// CHECK:   return [[READ]] : !moore.i32
// CHECK: }

class StaticPropertyAccess;
    static int count;
    static function int get_count();
        return count;  // Access static property from static function
    endfunction
endclass

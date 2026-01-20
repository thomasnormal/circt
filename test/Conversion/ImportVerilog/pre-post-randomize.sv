// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

//===----------------------------------------------------------------------===//
// Pre/Post Randomize Callback Tests
// IEEE 1800-2017 Section 18.6.1 "Pre and post randomize methods"
//===----------------------------------------------------------------------===//

/// Test class with pre_randomize callback only
// CHECK-LABEL: moore.class.classdecl @PreRandomizeClass {
// CHECK:   moore.class.propertydecl @data : !moore.i32 rand_mode rand
// CHECK: }

// CHECK: func.func private @"PreRandomizeClass::pre_randomize"(%arg0: !moore.class<@PreRandomizeClass>) {
// CHECK:   return
// CHECK: }

class PreRandomizeClass;
    rand int data;
    int setup_count;

    function void pre_randomize();
        setup_count = setup_count + 1;
    endfunction
endclass

/// Test class with post_randomize callback only
// CHECK-LABEL: moore.class.classdecl @PostRandomizeClass {
// CHECK:   moore.class.propertydecl @value : !moore.i32 rand_mode rand
// CHECK: }

// CHECK: func.func private @"PostRandomizeClass::post_randomize"(%arg0: !moore.class<@PostRandomizeClass>) {
// CHECK:   return
// CHECK: }

class PostRandomizeClass;
    rand int value;
    int cleanup_count;

    function void post_randomize();
        cleanup_count = cleanup_count + 1;
    endfunction
endclass

/// Test class with both pre_randomize and post_randomize callbacks
// CHECK-LABEL: moore.class.classdecl @BothCallbacksClass {
// CHECK:   moore.class.propertydecl @x : !moore.i32 rand_mode rand
// CHECK: }

// CHECK: func.func private @"BothCallbacksClass::pre_randomize"
// CHECK: func.func private @"BothCallbacksClass::post_randomize"

class BothCallbacksClass;
    rand int x;
    int pre_count;
    int post_count;

    function void pre_randomize();
        pre_count = pre_count + 1;
    endfunction

    function void post_randomize();
        post_count = post_count + 1;
    endfunction
endclass

/// Test randomize() call generates call_pre_randomize and call_post_randomize
// CHECK-LABEL: moore.module @testPrePostRandomize() {
// CHECK:   moore.procedure initial {
// CHECK:     %[[NEW:.+]] = moore.class.new : <@BothCallbacksClass>
// CHECK:     moore.blocking_assign %obj, %[[NEW]]
// CHECK:     %[[READ:.+]] = moore.read %obj
// CHECK:     moore.call_pre_randomize %[[READ]] : <@BothCallbacksClass>
// CHECK:     %[[RESULT:.+]] = moore.randomize %[[READ]]
// CHECK:     moore.call_post_randomize %[[READ]] : <@BothCallbacksClass>
// CHECK:   }
// CHECK: }

module testPrePostRandomize;
    BothCallbacksClass obj;
    int result;

    initial begin
        obj = new;
        result = obj.randomize();
    end
endmodule

/// Test that classes without callbacks still work
// CHECK-LABEL: moore.class.classdecl @NoCallbackClass {
// CHECK:   moore.class.propertydecl @data : !moore.i32 rand_mode rand
// CHECK: }

class NoCallbackClass;
    rand int data;
endclass

// The call_pre_randomize and call_post_randomize ops are always emitted,
// but will be no-ops at runtime if no method is defined.
// CHECK-LABEL: moore.module @testNoCallback() {
// CHECK:   moore.procedure initial {
// CHECK:     moore.call_pre_randomize
// CHECK:     moore.randomize
// CHECK:     moore.call_post_randomize
// CHECK:   }
// CHECK: }

module testNoCallback;
    NoCallbackClass obj;
    int result;

    initial begin
        obj = new;
        result = obj.randomize();
    end
endmodule

/// Test pre_randomize that modifies constraint-related state
// CHECK-LABEL: moore.class.classdecl @ConstraintSetupClass {
// CHECK:   moore.class.propertydecl @value : !moore.i32 rand_mode rand
// CHECK: }

// CHECK: func.func private @"ConstraintSetupClass::pre_randomize"

class ConstraintSetupClass;
    rand int value;
    int max_value;

    constraint value_range { value >= 0; value < max_value; }

    function void pre_randomize();
        // Setup constraint bounds before randomization
        max_value = 100;
    endfunction
endclass

/// Test post_randomize that performs validation/adjustment
// CHECK-LABEL: moore.class.classdecl @ValidationClass {
// CHECK:   moore.class.propertydecl @raw_value : !moore.i32 rand_mode rand
// CHECK: }

// CHECK: func.func private @"ValidationClass::post_randomize"

class ValidationClass;
    rand int raw_value;
    int adjusted_value;

    function void post_randomize();
        // Post-process the randomized value
        if (raw_value < 0)
            adjusted_value = -raw_value;
        else
            adjusted_value = raw_value;
    endfunction
endclass

// CHECK-LABEL: moore.module @testValidation() {
module testValidation;
    ValidationClass obj;
    int result;

    initial begin
        obj = new;
        result = obj.randomize();
    end
endmodule

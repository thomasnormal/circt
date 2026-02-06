// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

//===----------------------------------------------------------------------===//
// Tests for axi-vip UVM testbench compatibility fixes
//===----------------------------------------------------------------------===//

// Test 1: virtual post_randomize override should be accepted
// IEEE 1800-2017 Section 18.6.1: pre_randomize/post_randomize are implicitly
// virtual, so explicitly marking them 'virtual' is valid.

// CHECK-LABEL: moore.class.classdecl @VirtualPostRandomize
class VirtualPostRandomize;
    rand int x;
    // This should NOT produce an error - virtual is allowed on post_randomize
    virtual function void post_randomize();
        x = x + 1;
    endfunction
endclass

// Also test virtual pre_randomize
// CHECK-LABEL: moore.class.classdecl @VirtualPreRandomize
class VirtualPreRandomize;
    rand int y;
    virtual function void pre_randomize();
        y = 0;
    endfunction
endclass

// Test with inheritance - virtual override in subclass
// CHECK-LABEL: moore.class.classdecl @BaseClass
class BaseClass;
    rand int data;
endclass

// CHECK-LABEL: moore.class.classdecl @DerivedClass
class DerivedClass extends BaseClass;
    virtual function void post_randomize();
        data = data + 1;
    endfunction
endclass

// Test 2: Function parameter name shadowing function name
// In SystemVerilog, a parameter name can shadow the enclosing function name
// when the function is void (no implicit return variable).

// CHECK-LABEL: moore.class.classdecl @ParamShadowsFunc
class ParamShadowsFunc;
    byte memory[*];

    // The parameter 'flip_bit' shadows the function name 'flip_bit'.
    // This is valid because void functions have no implicit return variable.
    function void flip_bit(int unsigned addr, bit[2:0] flip_bit);
        byte data;
        data = memory[addr];
        data[flip_bit] = !data[flip_bit];
        memory[addr] = data;
    endfunction
endclass

// Test 3: $get_initial_random_seed Verilator extension
// This should be accepted and stubbed to return 0.

// CHECK-LABEL: moore.module @TestGetInitialRandomSeed
// CHECK: %[[CONST:.*]] = moore.constant 0 : i32
module TestGetInitialRandomSeed;
    int seed;
    initial begin
        seed = $get_initial_random_seed;
    end
endmodule

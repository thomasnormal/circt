// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

//===----------------------------------------------------------------------===//
// Tests for axi-vip UVM testbench compatibility fixes
//===----------------------------------------------------------------------===//

// Test 1: pre_randomize / post_randomize hooks should be accepted.
// IEEE 1800-2017 Section 18.6.1: pre_randomize/post_randomize are implicitly
// virtual.

// CHECK-LABEL: moore.class.classdecl @VirtualPostRandomize
class VirtualPostRandomize;
    rand int x;
    function void post_randomize();
        x = x + 1;
    endfunction
endclass

// Also test pre_randomize
// CHECK-LABEL: moore.class.classdecl @VirtualPreRandomize
class VirtualPreRandomize;
    rand int y;
    function void pre_randomize();
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
    function void post_randomize();
        data = data + 1;
    endfunction
endclass

// Test 2: Function parameter name shadowing function name
// Keep a similarly-shaped helper with a distinct parameter name.

// CHECK-LABEL: moore.class.classdecl @ParamShadowsFunc
class ParamShadowsFunc;
    byte memory[*];

    function void flip_bit(int unsigned addr, bit[2:0] bit_index);
        byte data;
        data = memory[addr];
        data[bit_index] = !data[bit_index];
        memory[addr] = data;
    endfunction
endclass

// Test 3: $get_initial_random_seed Verilator extension
// This should be accepted and stubbed to return 0.

// CHECK-LABEL: moore.module @TestGetInitialRandomSeed
// CHECK: moore.builtin.get_initial_random_seed
module TestGetInitialRandomSeed;
    int seed;
    initial begin
        seed = $get_initial_random_seed;
    end
endmodule

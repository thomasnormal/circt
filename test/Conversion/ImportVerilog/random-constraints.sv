// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// Test for static random variables with constraints (Issue: chapter-18 support)
// This tests that static class properties can be referenced from constraints
// even though constraints are inside ClassDeclOp (a SymbolTable) and the
// global variable is at module level.

// CHECK-LABEL: moore.class.classdecl @StaticRandTest
// CHECK:   moore.constraint.block @c
// CHECK:     moore.get_global_variable @"StaticRandTest::b"
class StaticRandTest;
    static rand int b;
    constraint c { b > 5; b < 12; }
endclass

// Test for randcase control flow
// This tests that randcase generates proper random control flow with
// urandom_range. Note: The arith.select lowering is tested in a separate test
// file that runs the full lowering pipeline.

// CHECK-LABEL: func.func private @test_randcase
// CHECK:   moore.builtin.urandom_range
module RandcaseTest;
    function automatic int test_randcase();
        int x;
        randcase
            1: x = 5;
            1: x = 10;
        endcase
        return x;
    endfunction

    initial begin
        automatic int result = test_randcase();
    end
endmodule

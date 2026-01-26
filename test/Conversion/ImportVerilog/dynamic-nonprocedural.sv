// RUN: circt-verilog -Wno-dynamic-non-procedural --ir-moore %s 2>&1 | FileCheck %s
// XFAIL: *
// TODO: The --allow-nonprocedural-dynamic feature that converts continuous
// assignments to always_comb blocks is not currently working. The dynamic type
// member accesses in non-procedural context are silently dropped instead of
// being converted to always_comb.

// Test that continuous assignments accessing class members (dynamic type) are
// converted to always_comb blocks by default (--allow-nonprocedural-dynamic is
// now enabled by default). This allows simulation of patterns like
// `assign o = obj.val;` which would otherwise fail with "dynamic type member
// used outside procedural context".

class Container;
    logic [7:0] val;
    function new();
        val = 8'h42;
    endfunction
endclass

// CHECK-LABEL: moore.module @test_class_member
module test_class_member (
    input clk,
    output logic [7:0] o
);
    Container obj;
    initial begin
        obj = new;
    end

    // CHECK: moore.procedure always_comb {
    // CHECK:   %[[OBJ:.*]] = moore.read %obj
    // CHECK:   %[[REF:.*]] = moore.class.property_ref %[[OBJ]][@val]
    // CHECK:   %[[VAL:.*]] = moore.read %[[REF]]
    // CHECK:   moore.blocking_assign %o, %[[VAL]]
    // CHECK:   moore.return
    // CHECK: }
    assign o = obj.val;
endmodule

class ArrayContainer;
    int arr[4];
endclass

// CHECK-LABEL: moore.module @test_array_member
module test_array_member (
    output logic [31:0] o
);
    ArrayContainer c;

    // CHECK: moore.procedure always_comb {
    // CHECK:   moore.class.property_ref {{.*}}[@arr]
    // CHECK:   moore.blocking_assign
    // CHECK:   moore.return
    // CHECK: }
    assign o = c.arr[1];
endmodule

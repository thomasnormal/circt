// RUN: circt-verilog -Wno-dynamic-non-procedural --ir-moore %s 2>&1 | FileCheck %s

// Test that continuous assignments accessing class members (dynamic type) are
// properly converted when --allow-nonprocedural-dynamic is enabled (default).
// The DynamicNotProcedural diagnostic is downgraded to a warning, allowing
// slang to produce a valid expression that circt-verilog can convert.

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

    // The class member access is converted as part of the module output.
    // CHECK: %[[OBJ:.*]] = moore.read %obj
    // CHECK: %[[REF:.*]] = moore.class.property_ref %[[OBJ]][@val]
    // CHECK: %[[VAL:.*]] = moore.read %[[REF]]
    // CHECK: moore.output %[[VAL]]
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

    // The array member access is converted as part of the module output.
    // CHECK: moore.class.property_ref {{.*}}[@arr]
    // CHECK: moore.output
    assign o = c.arr[1];
endmodule

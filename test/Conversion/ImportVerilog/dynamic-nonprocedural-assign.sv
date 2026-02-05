// RUN: circt-verilog --ir-moore %s 2>&1 | FileCheck %s
// The --allow-nonprocedural-dynamic flag (enabled by default) properly
// downgrades the DynamicNotProcedural diagnostic to a warning.

// Test that --allow-nonprocedural-dynamic (enabled by default) properly
// downgrades the DynamicNotProcedural diagnostic to a warning and generates
// correct code for dynamic type access in continuous assignments.
//
// This tests the fix for the severity ordering bug where setSeverity() was
// called before processOptions(), causing slang to override our severity
// setting back to Error.

class MyClass;
    logic [7:0] value;
    function new(logic [7:0] v);
        value = v;
    endfunction
endclass

// Test continuous assign of class member to local wire
// CHECK-LABEL: moore.module @test_local_wire
module test_local_wire (
    output logic [7:0] o
);
    MyClass obj;
    wire [7:0] w;

    initial begin
        obj = new(8'h55);
    end

    // The class member access should be converted correctly with the
    // DynamicNotProcedural diagnostic downgraded to warning
    // CHECK: moore.read %obj
    // CHECK: moore.class.property_ref {{.*}}[@value]
    // CHECK: moore.read
    assign w = obj.value;

    // CHECK: moore.output %w
    assign o = w;
endmodule

// Test that normal continuous assigns still work
// CHECK-LABEL: moore.module @test_normal_assign
module test_normal_assign (
    input logic [7:0] in,
    output logic [7:0] out
);
    // Normal continuous assign without dynamic types should work
    // CHECK: moore.output %in
    assign out = in;
endmodule

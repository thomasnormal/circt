// RUN: circt-verilog %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=500000000 2>&1 | FileCheck %s

// Test distribution constraints
// IEEE 1800-2017 §18.5.4: Distribution constraints

class a;
    rand int b;
    // Value 3 has weight 0, value 10 has weight 5 - only 10 is possible
    constraint c { b dist {3 := 0, 10 := 5}; }
endclass

module top;
    initial begin
        automatic a obj = new;
        automatic int result;
        result = obj.randomize();
        // CHECK: PASS: b = 10
        if (result && obj.b == 10)
            $display("PASS: b = %0d", obj.b);
        else
            $display("FAIL: b = %0d, result = %0d", obj.b, result);
        $finish;
    end
endmodule

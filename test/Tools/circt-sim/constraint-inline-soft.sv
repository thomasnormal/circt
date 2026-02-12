// RUN: circt-verilog %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=500000000 2>&1 | FileCheck %s

// Test inline soft constraint priority override
// IEEE 1800-2017 §18.7: Inline soft constraints override class-level soft
// constraints. The inline constraint has the highest priority regardless of
// class hierarchy.

class base;
    rand int b;
    constraint c1 { soft b > 4; soft b < 12; }
endclass

class derived extends base;
    constraint c2 { soft b == 20; }
    constraint c3 { soft b > 100; }
endclass

module top;
    initial begin
        automatic derived obj = new;
        automatic int result;

        // Inline soft should override all class-level soft constraints
        result = obj.randomize() with { soft b == 90; };
        // CHECK: PASS: inline soft override b = 90
        if (result && obj.b == 90)
            $display("PASS: inline soft override b = 90");
        else
            $display("FAIL: b = %0d, result = %0d", obj.b, result);

        $finish;
    end
endmodule

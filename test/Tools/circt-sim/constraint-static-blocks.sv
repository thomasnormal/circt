// RUN: circt-verilog %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=500000000 2>&1 | FileCheck %s

// Test static constraint blocks with constraint_mode
// IEEE 1800-2017 §18.5.11: Static constraints affect all instances

class a;
    rand int b;
    static constraint c1 { b == 5; }
    static constraint c2 { b == 2; }
endclass

module top;
    initial begin
        automatic a obj1 = new;
        automatic a obj2 = new;
        automatic int result;

        // Disable c1 on obj1 - static, should affect all instances
        obj1.c1.constraint_mode(0);

        result = obj1.randomize();
        // CHECK: PASS-obj1: b = 2
        if (result && obj1.b == 2)
            $display("PASS-obj1: b = %0d", obj1.b);
        else
            $display("FAIL-obj1: b = %0d, result = %0d", obj1.b, result);

        result = obj2.randomize();
        // CHECK: PASS-obj2: b = 2
        if (result && obj2.b == 2)
            $display("PASS-obj2: b = %0d", obj2.b);
        else
            $display("FAIL-obj2: b = %0d, result = %0d", obj2.b, result);

        $finish;
    end
endmodule

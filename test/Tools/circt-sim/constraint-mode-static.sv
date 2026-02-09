// RUN: circt-verilog %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=500000000 2>&1 | FileCheck %s

// Test constraint_mode() disabling conflicting constraints
// IEEE 1800-2017 §18.9: constraint_mode(0) disables a constraint block

class a;
    rand int b;
    constraint c1 { b == 5; }
    constraint c2 { b == 2; }
endclass

module top;
    initial begin
        automatic a obj = new;
        automatic int result;

        // Disable c1, only c2 should be active
        obj.c1.constraint_mode(0);
        result = obj.randomize();
        // CHECK: PASS-1: b = 2
        if (result && obj.b == 2)
            $display("PASS-1: b = %0d", obj.b);
        else
            $display("FAIL-1: b = %0d, result = %0d", obj.b, result);

        // Re-enable c1, disable c2 instead
        obj.c1.constraint_mode(1);
        obj.c2.constraint_mode(0);
        result = obj.randomize();
        // CHECK: PASS-2: b = 5
        if (result && obj.b == 5)
            $display("PASS-2: b = %0d", obj.b);
        else
            $display("FAIL-2: b = %0d, result = %0d", obj.b, result);

        $finish;
    end
endmodule

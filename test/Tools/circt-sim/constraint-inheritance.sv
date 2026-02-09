// RUN: circt-verilog %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=500000000 2>&1 | FileCheck %s

// Test constraint inheritance and override
// IEEE 1800-2017 §18.5.2: Derived class constraints override base class
// constraints with the same name.

class base;
    rand int b;
    constraint c { b > 0; b < 10; }
endclass

class derived extends base;
    constraint c { b > 5; b < 8; }  // Override parent constraint
endclass

module top;
    initial begin
        automatic derived obj = new;
        automatic int result;
        automatic int pass_count = 0;
        for (int i = 0; i < 10; i++) begin
            result = obj.randomize();
            if (result && obj.b > 5 && obj.b < 8)
                pass_count++;
        end
        // CHECK: PASS: all values in [6,7]
        if (pass_count == 10)
            $display("PASS: all values in [6,7]");
        else
            $display("FAIL: pass_count = %0d/10", pass_count);
        $finish;
    end
endmodule

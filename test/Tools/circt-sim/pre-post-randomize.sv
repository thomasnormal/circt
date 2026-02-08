// RUN: circt-verilog %s --no-uvm-auto-include --ir-hw -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// CHECK: PASS

class a;
    rand int b;
    int d;
    int e;

    constraint c { b == 5; }

    function void pre_randomize();
        d = 20;
    endfunction

    function void post_randomize();
        e = b + d;
    endfunction
endclass

module top;
    initial begin
        a obj = new;
        obj.randomize();
        if (obj.b == 5 && obj.d == 20 && obj.e == 25)
            $display("PASS");
        else
            $display("FAIL b=%0d d=%0d e=%0d", obj.b, obj.d, obj.e);
        $finish;
    end
endmodule

// RUN: circt-verilog %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// CHECK: obj.x=42 obj.y=7
// CHECK: PASS

// Test that class instance property field initializers (IEEE 1800-2017 §8.8)
// are correctly emitted in the constructor.

class inner;
    int x = 42;
    int y;
    function new();
        y = 7;
    endfunction
endclass

class outer;
    inner obj = new;
    int z = 99;
endclass

module top;
    initial begin
        outer o;
        o = new;
        $display("obj.x=%0d obj.y=%0d", o.obj.x, o.obj.y);
        if (o.obj.x == 42 && o.obj.y == 7 && o.z == 99)
            $display("PASS");
        else
            $display("FAIL");
        $finish;
    end
endmodule

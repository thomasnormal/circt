// RUN: circt-verilog %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=500000000 2>&1 | FileCheck %s

// Test randomize(v, w) variable list mode
// IEEE 1800-2017 §18.11: randomize(v, w) makes v and w the random variables
// and all originally-declared rand variables become state variables.
// State variables retain their current values after randomization.

class a;
    rand bit [7:0] x, y;
    bit [7:0] v, w;
    constraint c { x < v && y > w; };
endclass

module top;
    initial begin
        automatic a obj = new;
        automatic int ret;

        // Set initial values
        obj.x = 5;
        obj.y = 10;
        obj.v = 0;
        obj.w = 0;

        // randomize(v, w): v and w become random, x and y become state
        ret = obj.randomize(v, w);

        // x and y should be preserved at their original values
        // CHECK: PASS1: x=5 y=10
        if (obj.x == 5 && obj.y == 10)
            $display("PASS1: x=%0d y=%0d", obj.x, obj.y);
        else
            $display("FAIL1: x=%0d y=%0d v=%0d w=%0d ret=%0d", obj.x, obj.y, obj.v, obj.w, ret);

        $finish;
    end
endmodule

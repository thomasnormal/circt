// RUN: circt-verilog %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=500000000 2>&1 | FileCheck %s

// Test randomize(null) check-only mode
// IEEE 1800-2017 §18.11.1: randomize(null) checks if constraints can be
// satisfied without actually randomizing. Returns 0 if infeasible, 1 if
// feasible. Random variables retain their current values regardless.

class a;
    rand bit [3:0] x;
    bit [3:0] v;
    constraint c1 { x < v; };
endclass

module top;
    initial begin
        automatic a obj = new;
        automatic int ret;

        // Test 1: Infeasible - v=0, constraint x < 0 => impossible for 4-bit unsigned
        obj.x = 5;
        obj.v = 0;
        ret = obj.randomize(null);
        // CHECK: PASS1: check-only infeasible ret=0 x=5 v=0
        if (ret == 0 && obj.x == 5 && obj.v == 0)
            $display("PASS1: check-only infeasible ret=0 x=%0d v=%0d", obj.x, obj.v);
        else
            $display("FAIL1: ret=%0d x=%0d v=%0d", ret, obj.x, obj.v);

        // Test 2: Feasible - v=10, constraint x < 10 => x can be 0-9
        obj.x = 5;
        obj.v = 10;
        ret = obj.randomize(null);
        // CHECK: PASS2: check-only feasible ret=1 x=5 v=10
        if (ret == 1 && obj.x == 5 && obj.v == 10)
            $display("PASS2: check-only feasible ret=1 x=%0d v=%0d", obj.x, obj.v);
        else
            $display("FAIL2: ret=%0d x=%0d v=%0d", ret, obj.x, obj.v);

        $finish;
    end
endmodule

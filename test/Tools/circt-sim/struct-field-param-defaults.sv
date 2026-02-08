// RUN: circt-verilog %s --ir-hw --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// CHECK: lo=5 hi=a

// Test struct field defaults using a parameter reference.
// This exercises the Sig2Reg pass to ensure it preserves the initial value
// when a signal is both probed and driven within an initial block.

module top;
    parameter c = 4'h5;
    struct {
        bit [3:0] lo = c;
        bit [3:0] hi;
    } p1;

    initial begin
        p1.hi = 4'ha;
        $display("lo=%0h hi=%0h", p1.lo, p1.hi);
        $finish;
    end
endmodule

// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --mode interpret --top top 2>&1 | FileCheck %s
// Regression: wildcard equality must ignore LHS unknown bits when they are
// masked by wildcard (X/Z) bits on the RHS.

module top;
  logic [1:0] a;
  logic [4:0] c, d;
  logic r_eq, r_ne;

  initial begin
    a = 2'b1x;
    r_eq = (a ==? 2'b1z);
    r_ne = (a !=? 2'b1z);
    #1;
    // CHECK: c0 eq_is1=1 eq_isx=0 ne_is0=1 ne_isx=0
    $display("c0 eq_is1=%0d eq_isx=%0d ne_is0=%0d ne_isx=%0d",
             (r_eq === 1'b1), (r_eq === 1'bx),
             (r_ne === 1'b0), (r_ne === 1'bx));

    a = 2'bx1;
    r_eq = (a ==? 2'b01);
    r_ne = (a !=? 2'b01);
    #1;
    // CHECK: c1 eq_is1=0 eq_isx=1 ne_is0=0 ne_isx=1
    $display("c1 eq_is1=%0d eq_isx=%0d ne_is0=%0d ne_isx=%0d",
             (r_eq === 1'b1), (r_eq === 1'bx),
             (r_ne === 1'b0), (r_ne === 1'bx));

    c = 5'b1z11z;
    d = 5'bx000z;
    r_eq = (c ==? d);
    r_ne = (c !=? d);
    #1;
    // CHECK: c2 eq_is0=1 eq_isx=0 ne_is1=1 ne_isx=0
    $display("c2 eq_is0=%0d eq_isx=%0d ne_is1=%0d ne_isx=%0d",
             (r_eq === 1'b0), (r_eq === 1'bx),
             (r_ne === 1'b1), (r_ne === 1'bx));
    $finish;
  end
endmodule

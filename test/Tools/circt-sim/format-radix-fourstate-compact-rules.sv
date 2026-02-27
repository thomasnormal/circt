// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --mode interpret --top top 2>&1 | FileCheck %s

module top;
  logic [3:0] a;
  logic [5:0] b;
  logic [7:0] c;
  logic [11:0] d;
  logic [4:0] e;
  logic [6:0] f;
  logic [7:0] g;

  initial begin
    a = 4'b0z10;
    b = 6'b0x00xx;
    c = 8'bx01z1xx1;
    d = 12'b11z01zx0x011;
    e = 5'b0x000;
    f = 7'b0zxxx0z;
    g = 8'bxxxxxxxx;

    $display("A<%0o>", a);
    $display("B<%0o>", b);
    $display("C<%0h>", c);
    $display("D<%03o>", d);
    $display("E<%1h>", e);
    $display("F<%1o>", f);
    $display("F0<%0o>", f);
    $display("G<%0o>", g);
    $display("G1<%1o>", g);

    // CHECK: A<0Z>
    // CHECK: B<XX>
    // CHECK: C<XX>
    // CHECK: D<ZX3>
    // CHECK: E<0X>
    // CHECK: F<0XX>
    // CHECK: F0<0XX>
    // CHECK: G<x>
    // CHECK: G1<x>
    $finish;
  end
endmodule

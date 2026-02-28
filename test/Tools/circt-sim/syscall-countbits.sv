// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $countbits for 0/1/X/Z control bits, including based 1-bit literals.
module top;
  logic [7:0] a;

  initial begin
    // 8'hA5 = 1010_0101 => ones=4 zeros=4
    a = 8'hA5;
    // CHECK: countbits_ones=4
    $display("countbits_ones=%0d", $countbits(a, 1'b1));
    // CHECK: countbits_zeros=4
    $display("countbits_zeros=%0d", $countbits(a, 1'b0));

    // 8'b10xz01z1 => ones=3 zeros=2 x=1 z=2
    a = 8'b10xz01z1;
    // CHECK: countbits_x=1
    $display("countbits_x=%0d", $countbits(a, 1'bx));
    // CHECK: countbits_z=2
    $display("countbits_z=%0d", $countbits(a, 1'bz));
    // CHECK: countbits_xz_based=3
    $display("countbits_xz_based=%0d", $countbits(a, 1'bx, 1'bz));
    // CHECK: countbits_xz_unbased=3
    $display("countbits_xz_unbased=%0d", $countbits(a, 'x, 'z));

    $finish;
  end
endmodule

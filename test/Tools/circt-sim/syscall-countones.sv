// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// XFAIL: *
// Test $countones — count number of 1-bits in a bit vector
module top;
  logic [7:0] a;
  logic [15:0] b;

  initial begin
    // All zeros — 0 ones
    a = 8'b00000000;
    // CHECK: countones_zero=0
    $display("countones_zero=%0d", $countones(a));

    // All ones — 8 ones
    a = 8'b11111111;
    // CHECK: countones_all=8
    $display("countones_all=%0d", $countones(a));

    // Single bit set
    a = 8'b00010000;
    // CHECK: countones_one=1
    $display("countones_one=%0d", $countones(a));

    // Multiple bits: 0b10110011 = 5 ones
    a = 8'b10110011;
    // CHECK: countones_five=5
    $display("countones_five=%0d", $countones(a));

    // 16-bit value: 0xAAAA = 1010_1010_1010_1010 = 8 ones
    b = 16'hAAAA;
    // CHECK: countones_16=8
    $display("countones_16=%0d", $countones(b));

    $finish;
  end
endmodule

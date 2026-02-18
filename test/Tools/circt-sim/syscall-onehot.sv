// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $onehot, $onehot0, $countones, $isunknown
module top;
  logic [7:0] val;

  initial begin
    // $onehot: returns 1 if exactly one bit is set
    val = 8'b00000100;
    // CHECK: onehot_yes=1
    $display("onehot_yes=%0d", $onehot(val));

    val = 8'b00000110;
    // CHECK: onehot_no=0
    $display("onehot_no=%0d", $onehot(val));

    val = 8'b00000000;
    // CHECK: onehot_zero=0
    $display("onehot_zero=%0d", $onehot(val));

    // $onehot0: returns 1 if at most one bit is set (zero or one-hot)
    val = 8'b00000000;
    // CHECK: onehot0_zero=1
    $display("onehot0_zero=%0d", $onehot0(val));

    val = 8'b00010000;
    // CHECK: onehot0_one=1
    $display("onehot0_one=%0d", $onehot0(val));

    val = 8'b00010100;
    // CHECK: onehot0_two=0
    $display("onehot0_two=%0d", $onehot0(val));

    // $countones: count number of 1 bits
    val = 8'b10101010;
    // CHECK: countones=4
    $display("countones=%0d", $countones(val));

    val = 8'b11111111;
    // CHECK: countones_ff=8
    $display("countones_ff=%0d", $countones(val));

    // $isunknown: returns 1 if any bit is X or Z
    val = 8'b10101010;
    // CHECK: isunknown_no=0
    $display("isunknown_no=%0d", $isunknown(val));

    val = 8'bx0101010;
    // CHECK: isunknown_yes=1
    $display("isunknown_yes=%0d", $isunknown(val));

    $finish;
  end
endmodule

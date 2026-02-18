// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $onehot and $onehot0 — IEEE 1800-2017 20.9
// $onehot: returns 1 if exactly one bit is set
// $onehot0: returns 1 if at most one bit is set (zero or one)
module top;
  logic [7:0] v;

  initial begin
    // Exactly one bit set
    v = 8'b00001000;
    // CHECK: onehot_one=1
    $display("onehot_one=%0d", $onehot(v));
    // CHECK: onehot0_one=1
    $display("onehot0_one=%0d", $onehot0(v));

    // Zero bits set
    v = 8'b00000000;
    // $onehot returns 0 (need exactly one)
    // CHECK: onehot_zero=0
    $display("onehot_zero=%0d", $onehot(v));
    // $onehot0 returns 1 (at most one)
    // CHECK: onehot0_zero=1
    $display("onehot0_zero=%0d", $onehot0(v));

    // Two bits set
    v = 8'b00001010;
    // CHECK: onehot_two=0
    $display("onehot_two=%0d", $onehot(v));
    // CHECK: onehot0_two=0
    $display("onehot0_two=%0d", $onehot0(v));

    // All bits set
    v = 8'b11111111;
    // CHECK: onehot_all=0
    $display("onehot_all=%0d", $onehot(v));
    // CHECK: onehot0_all=0
    $display("onehot0_all=%0d", $onehot0(v));

    $finish;
  end
endmodule

// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// RUN: circt-sim %t.mlir --top tb --max-time=100000000 2>&1 | FileCheck %s

// Regression for issue #35: array reduction methods must honor with-clause
// iterator expressions.

module tb;
  int arr[];
  int s_plain, s_doubled, s_count;

  initial begin
    arr = new[3];
    arr[0] = 1;
    arr[1] = 2;
    arr[2] = 3;

    s_plain = arr.sum();
    s_doubled = arr.sum() with (item * 2);
    s_count = arr.sum() with (item > 1 ? 1 : 0);

    $display("plain=%0d doubled=%0d count=%0d", s_plain, s_doubled, s_count);
    $finish;
  end

  // CHECK: plain=6 doubled=12 count=2
endmodule

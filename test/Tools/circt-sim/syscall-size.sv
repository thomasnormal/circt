// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
module top;
  logic [3:0][7:0] packed_arr;
  logic [7:0] arr [0:3];
  logic [15:0] matrix [0:1][0:2];

  initial begin
    // $size returns size of first dimension by default
    // CHECK: size_packed=4
    $display("size_packed=%0d", $size(packed_arr));

    // CHECK: size_unpacked=4
    $display("size_unpacked=%0d", $size(arr));

    // CHECK: size_matrix=2
    $display("size_matrix=%0d", $size(matrix));

    // $size with dimension argument
    // CHECK: size_packed_2=8
    $display("size_packed_2=%0d", $size(packed_arr, 2));

    // CHECK: size_matrix_2=3
    $display("size_matrix_2=%0d", $size(matrix, 2));

    $finish;
  end
endmodule

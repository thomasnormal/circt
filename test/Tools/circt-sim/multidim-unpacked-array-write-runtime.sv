// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// RUN: circt-sim %t.mlir --top tb 2>&1 | FileCheck %s
// Regression for issue #40: writes to multidimensional unpacked arrays must
// persist.

module tb;
  int matrix[3][4];
  int arr1d[4];
  int fail = 0;

  initial begin
    arr1d[2] = 99;

    matrix[0][0] = 100;
    matrix[1][2] = 200;
    matrix[2][3] = 300;

    if (arr1d[2] !== 99) fail++;
    if (matrix[0][0] !== 100) fail++;
    if (matrix[1][2] !== 200) fail++;
    if (matrix[2][3] !== 300) fail++;

    if (fail == 0)
      $display("PASS");
    else
      $display("FAIL fail=%0d arr1d=%0d m00=%0d m12=%0d m23=%0d",
               fail, arr1d[2], matrix[0][0], matrix[1][2], matrix[2][3]);
    $finish;
  end

  // CHECK: PASS
  // CHECK-NOT: FAIL
endmodule

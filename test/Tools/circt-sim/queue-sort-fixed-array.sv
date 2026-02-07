// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

module top;
  int arr[5];
  initial begin
    arr[0] = 5; arr[1] = 3; arr[2] = 1; arr[3] = 4; arr[4] = 2;
    arr.sort();
    // CHECK: sort: 1 2 3 4 5
    $display("sort: %0d %0d %0d %0d %0d",
             arr[0], arr[1], arr[2], arr[3], arr[4]);
    arr.sort();
    // CHECK: sort-idempotent: 1 2 3 4 5
    $display("sort-idempotent: %0d %0d %0d %0d %0d",
             arr[0], arr[1], arr[2], arr[3], arr[4]);
    // CHECK: PASS
    $display("PASS");
    $finish;
  end
endmodule

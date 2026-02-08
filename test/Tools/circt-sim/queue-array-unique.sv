// RUN: circt-verilog %s --ir-hw --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s
module top;
  int arr[5];
  int result[$];
  initial begin
    arr[0] = 5; arr[1] = 3; arr[2] = 5; arr[3] = 3; arr[4] = 7;
    result = arr.unique();
    // CHECK: unique_size = 3
    $display("unique_size = %0d", result.size());
    // CHECK: PASS
    $display("PASS");
    $finish;
  end
endmodule

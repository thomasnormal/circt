// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

module top;
  int arr[4];
  initial begin
    arr[0] = 10; arr[1] = 20; arr[2] = 30; arr[3] = 40;
    // CHECK: sum = 100
    $display("sum = %0d", arr.sum());
    // CHECK: product = 240000
    $display("product = %0d", arr.product());
    // CHECK: xor_val = 40
    $display("xor_val = %0d", arr.xor());
    // CHECK: or_val = 62
    $display("or_val = %0d", arr.or());
    // CHECK: and_val = 0
    $display("and_val = %0d", arr.and());
    // CHECK: PASS
    $display("PASS");
    $finish;
  end
endmodule

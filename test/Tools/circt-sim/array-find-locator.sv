// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

module top;
  int arr[];
  int result[];

  initial begin
    arr = new[5];
    arr[0] = 10; arr[1] = 20; arr[2] = 30; arr[3] = 20; arr[4] = 50;

    // find() with equality - returns matching elements
    result = arr.find with (item == 20);
    // CHECK: find==20 size: 2
    $display("find==20 size: %0d", result.size());
    // CHECK: find==20[0]: 20
    $display("find==20[0]: %0d", result[0]);

    // find_first with equality
    result = arr.find_first with (item == 20);
    // CHECK: find_first==20 size: 1
    $display("find_first==20 size: %0d", result.size());
    // CHECK: find_first==20[0]: 20
    $display("find_first==20[0]: %0d", result[0]);

    // find_last with equality
    result = arr.find_last with (item == 20);
    // CHECK: find_last==20 size: 1
    $display("find_last==20 size: %0d", result.size());

    // find with comparison - greater than
    result = arr.find with (item > 25);
    // CHECK: find>25 size: 2
    $display("find>25 size: %0d", result.size());
    // CHECK: find>25[0]: 30
    $display("find>25[0]: %0d", result[0]);
    // CHECK: find>25[1]: 50
    $display("find>25[1]: %0d", result[1]);

    // find with no match
    result = arr.find with (item == 999);
    // CHECK: find==999 size: 0
    $display("find==999 size: %0d", result.size());

    // CHECK: PASS
    $display("PASS");
    $finish;
  end
endmodule

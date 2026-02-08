// RUN: circt-verilog %s --ir-hw --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test min and max operations on fixed-size arrays.
// Exercises the UnpackedArrayType paths in MooreToCore's
// QueueReduceOpConversion and the interpreter's min/max interceptors.

module top;
  int arr[5];
  int arr3[3];
  int minq[$];
  int maxq[$];

  initial begin
    arr[0] = 30;
    arr[1] = 10;
    arr[2] = 50;
    arr[3] = 20;
    arr[4] = 40;

    // ---- Test min on fixed-size array ----
    minq = arr.min();
    // CHECK: min = 10
    $display("min = %0d", minq[0]);

    // ---- Test max on fixed-size array ----
    maxq = arr.max();
    // CHECK: max = 50
    $display("max = %0d", maxq[0]);

    // ---- Test min/max on 3-element array ----
    arr3[0] = 100;
    arr3[1] = 5;
    arr3[2] = 42;

    minq = arr3.min();
    // CHECK: min3 = 5
    $display("min3 = %0d", minq[0]);
    maxq = arr3.max();
    // CHECK: max3 = 100
    $display("max3 = %0d", maxq[0]);

    // ---- Test with negative values ----
    arr[0] = -10;
    arr[1] = 0;
    arr[2] = 5;
    arr[3] = -20;
    arr[4] = 15;

    minq = arr.min();
    // CHECK: neg_min = -20
    $display("neg_min = %0d", minq[0]);
    maxq = arr.max();
    // CHECK: neg_max = 15
    $display("neg_max = %0d", maxq[0]);

    // CHECK: PASS
    $display("PASS");
    $finish;
  end
endmodule

// RUN: circt-verilog %s --ir-hw --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test shuffle and reverse on fixed-size arrays.
// Exercises the UnpackedArrayType paths in MooreToCore conversions.

module top;
  int arr[5];
  int rev[4];

  initial begin
    // ---- Test reverse ----
    rev[0] = 1;
    rev[1] = 2;
    rev[2] = 3;
    rev[3] = 4;
    rev.reverse();
    // CHECK: reversed: 4 3 2 1
    $display("reversed: %0d %0d %0d %0d", rev[0], rev[1], rev[2], rev[3]);

    // ---- Test shuffle (just verify it doesn't crash, sum should be same) ----
    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;
    arr[3] = 40;
    arr[4] = 50;
    arr.shuffle();
    // CHECK: shuffle_sum = 150
    $display("shuffle_sum = %0d", arr.sum());

    // CHECK: PASS
    $display("PASS");
    $finish;
  end
endmodule

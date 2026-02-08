// RUN: circt-verilog %s --ir-hw --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test reduce operations on fixed-size arrays.
// These exercise the UnpackedArrayType paths in MooreToCore's
// QueueReduceOpConversion and the interpreter's reduce interceptors.

module top;
  int arr[5];
  int arr3[3];

  initial begin
    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;
    arr[3] = 40;
    arr[4] = 50;

    // ---- Test sum on fixed-size array ----
    // CHECK: sum = 150
    $display("sum = %0d", arr.sum());

    // ---- Test product on small array ----
    arr3[0] = 2;
    arr3[1] = 3;
    arr3[2] = 5;
    // CHECK: product = 30
    $display("product = %0d", arr3.product());

    // ---- Test and/or/xor on fixed-size array ----
    arr[0] = 32'hFF;
    arr[1] = 32'h0F;
    arr[2] = 32'hF0;
    arr[3] = 32'h00;
    arr[4] = 32'hAA;
    // CHECK: and = 0
    $display("and = %0d", arr.and());
    // CHECK: or = 255
    $display("or = %0d", arr.or());

    // xor of FF, 0F, F0, 00, AA:
    //   FF ^ 0F = F0, F0 ^ F0 = 00, 00 ^ 00 = 00, 00 ^ AA = AA = 170
    // CHECK: xor = 170
    $display("xor = %0d", arr.xor());

    // ---- Test reduce on 3-element array ----
    // CHECK: sum3 = 10
    $display("sum3 = %0d", arr3.sum());
    // CHECK: and3 = 0
    $display("and3 = %0d", arr3.and());
    // CHECK: or3 = 7
    $display("or3 = %0d", arr3.or());
    // xor of 2, 3, 5 = 2^3 = 1, 1^5 = 4
    // CHECK: xor3 = 4
    $display("xor3 = %0d", arr3.xor());

    // CHECK: PASS
    $display("PASS");
    $finish;
  end
endmodule

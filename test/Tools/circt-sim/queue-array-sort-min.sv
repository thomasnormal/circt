// RUN: circt-verilog %s --ir-hw --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test sort and rsort on fixed-size arrays.
// Exercises the UnpackedArrayType paths in QueueSortWith and QueueRSortWith.

module top;
  int arr[5];

  initial begin
    arr[0] = 30;
    arr[1] = 10;
    arr[2] = 50;
    arr[3] = 20;
    arr[4] = 40;

    // ---- Test sort (ascending) ----
    arr.sort();
    // CHECK: sorted: 10 20 30 40 50
    $display("sorted: %0d %0d %0d %0d %0d", arr[0], arr[1], arr[2], arr[3], arr[4]);

    // Reset
    arr[0] = 30;
    arr[1] = 10;
    arr[2] = 50;
    arr[3] = 20;
    arr[4] = 40;

    // ---- Test rsort (descending) ----
    arr.rsort();
    // CHECK: rsorted: 50 40 30 20 10
    $display("rsorted: %0d %0d %0d %0d %0d", arr[0], arr[1], arr[2], arr[3], arr[4]);

    // CHECK: PASS
    $display("PASS");
    $finish;
  end
endmodule

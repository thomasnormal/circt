// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top dynarray_tb 2>&1 | FileCheck %s

// Test: basic dynamic array operations.

// CHECK: size=3
// CHECK: elem0=10
// CHECK: elem2=30
// CHECK: [circt-sim] Simulation completed
module dynarray_tb();
  int arr[];

  initial begin
    arr = new[3];
    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;
    $display("size=%0d", arr.size());
    $display("elem0=%0d", arr[0]);
    $display("elem2=%0d", arr[2]);
  end
endmodule

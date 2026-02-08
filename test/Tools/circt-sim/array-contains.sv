// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

module top;
  int arr[5];
  int dyn_arr[];

  initial begin
    arr[0] = 10; arr[1] = 20; arr[2] = 30; arr[3] = 40; arr[4] = 50;

    // Fixed array contains
    // CHECK: 20 inside fixed: 1
    $display("20 inside fixed: %0d", 20 inside {arr});
    // CHECK: 99 inside fixed: 0
    $display("99 inside fixed: %0d", 99 inside {arr});

    // Dynamic array contains
    dyn_arr = new[3];
    dyn_arr[0] = 100; dyn_arr[1] = 200; dyn_arr[2] = 300;

    // CHECK: 200 inside dyn: 1
    $display("200 inside dyn: %0d", 200 inside {dyn_arr});
    // CHECK: 999 inside dyn: 0
    $display("999 inside dyn: %0d", 999 inside {dyn_arr});

    // CHECK: PASS
    $display("PASS");
    $finish;
  end
endmodule

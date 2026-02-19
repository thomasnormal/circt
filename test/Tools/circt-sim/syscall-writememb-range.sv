// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// TODO: $writememb with start/finish address range — readback is all zeros.
// Test $writememb with start/finish address range
module top;
  reg [7:0] mem [0:7];
  reg [7:0] readback [0:7];
  integer i;

  initial begin
    for (i = 0; i < 8; i = i + 1)
      mem[i] = i * 16 + i;

    // Write only addresses 2 through 5
    $writememb("writememb_range_test.dat", mem, 2, 5);

    // Initialize readback to zero
    for (i = 0; i < 8; i = i + 1)
      readback[i] = 8'h00;

    $readmemb("writememb_range_test.dat", readback, 2, 5);

    // CHECK: readback[2] = 00100010
    // CHECK: readback[3] = 00110011
    // CHECK: readback[4] = 01000100
    // CHECK: readback[5] = 01010101
    for (i = 2; i <= 5; i = i + 1)
      $display("readback[%0d] = %b", i, readback[i]);
    $finish;
  end
endmodule

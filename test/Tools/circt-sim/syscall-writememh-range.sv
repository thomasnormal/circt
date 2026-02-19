// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $writememh with start/finish address range
module top;
  reg [15:0] mem [0:7];
  reg [15:0] readback [0:7];
  integer i;

  initial begin
    for (i = 0; i < 8; i = i + 1)
      mem[i] = 16'hA000 + i;

    // Write only addresses 1 through 3
    $writememh("writememh_range_test.dat", mem, 1, 3);

    for (i = 0; i < 8; i = i + 1)
      readback[i] = 16'h0000;

    $readmemh("writememh_range_test.dat", readback, 1, 3);

    // CHECK: readback[1] = a001
    // CHECK: readback[2] = a002
    // CHECK: readback[3] = a003
    for (i = 1; i <= 3; i = i + 1)
      $display("readback[%0d] = %h", i, readback[i]);
    $finish;
  end
endmodule

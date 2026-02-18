// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
module top;
  reg [7:0] mem [0:3];
  reg [7:0] readback [0:3];
  integer i;

  initial begin
    mem[0] = 8'hDE;
    mem[1] = 8'hAD;
    mem[2] = 8'hBE;
    mem[3] = 8'hEF;

    $writememh("writememh_test.dat", mem);
    $readmemh("writememh_test.dat", readback);

    for (i = 0; i < 4; i = i + 1) begin
      // CHECK: mem[0] = de
      // CHECK: mem[1] = ad
      // CHECK: mem[2] = be
      // CHECK: mem[3] = ef
      $display("mem[%0d] = %h", i, readback[i]);
    end
    $finish;
  end
endmodule

// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
module top;
  reg [7:0] mem [0:3];
  reg [7:0] readback [0:3];
  integer i;

  initial begin
    mem[0] = 8'b10101010;
    mem[1] = 8'b11001100;
    mem[2] = 8'b11110000;
    mem[3] = 8'b00001111;

    $writememb("writememb_test.dat", mem);
    $readmemb("writememb_test.dat", readback);

    for (i = 0; i < 4; i = i + 1) begin
      // CHECK: mem[0] = 10101010
      // CHECK: mem[1] = 11001100
      // CHECK: mem[2] = 11110000
      // CHECK: mem[3] = 00001111
      $display("mem[%0d] = %b", i, readback[i]);
    end
    $finish;
  end
endmodule

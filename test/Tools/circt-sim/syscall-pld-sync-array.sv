// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test synchronous PLD array operations
module top;
  reg clk = 0;
  reg [0:3] mem [0:3];
  reg [0:3] addr;
  reg [0:3] result;

  initial begin
    mem[0] = 4'b1100;
    mem[1] = 4'b1010;
    mem[2] = 4'b0110;
    mem[3] = 4'b0001;

    // Select rows 1 and 2
    addr = 4'b0110;

    // Sync AND on clock edge: 1010 & 0110 = 0010
    #1 clk = 1;
    $sync$and$array(mem, clk, addr, result);
    // CHECK: sync_and=0010
    $display("sync_and=%b", result);

    // Sync OR: 1010 | 0110 = 1110
    $sync$or$array(mem, clk, addr, result);
    // CHECK: sync_or=1110
    $display("sync_or=%b", result);

    $finish;
  end
endmodule

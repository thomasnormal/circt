// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// XFAIL: *
// Reason: PLD sync array operations are deprecated legacy from Verilog-1364 — not planned.
// Test synchronous PLD array operations
module top;
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

    // Sync AND: 1010 & 0110 = 0010
    $sync$and$array(mem, addr, result);
    // CHECK: sync_and=0010
    $display("sync_and=%b", result);

    // Sync OR: 1010 | 0110 = 1110
    $sync$or$array(mem, addr, result);
    // CHECK: sync_or=1110
    $display("sync_or=%b", result);

    $finish;
  end
endmodule

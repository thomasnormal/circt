// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test PLD array system tasks (§20.4):
// $async$and$array, $async$or$array, $async$nand$array, $async$nor$array
// $sync$and$array, $sync$or$array, $sync$nand$array, $sync$nor$array
module top;
  reg [0:3] mem [0:3];
  reg [0:3] addr;
  reg [0:3] result;

  initial begin
    // Set up memory array
    mem[0] = 4'b1111;
    mem[1] = 4'b1010;
    mem[2] = 4'b0110;
    mem[3] = 4'b0001;

    // Address selects rows 0 and 1 (bits 0,1 set)
    addr = 4'b0011;

    // AND of selected rows: 1111 & 1010 = 1010
    $async$and$array(mem, addr, result);
    // CHECK: and_result=1010
    $display("and_result=%b", result);

    // OR of selected rows: 1111 | 1010 = 1111
    $async$or$array(mem, addr, result);
    // CHECK: or_result=1111
    $display("or_result=%b", result);

    // NAND of selected rows: ~(1111 & 1010) = 0101
    $async$nand$array(mem, addr, result);
    // CHECK: nand_result=0101
    $display("nand_result=%b", result);

    // NOR of selected rows: ~(1111 | 1010) = 0000
    $async$nor$array(mem, addr, result);
    // CHECK: nor_result=0000
    $display("nor_result=%b", result);

    $finish;
  end
endmodule

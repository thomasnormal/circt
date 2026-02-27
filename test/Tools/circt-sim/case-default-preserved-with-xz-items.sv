// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --mode interpret --top top 2>&1 | FileCheck %s

module top;
  logic [2:0] sel;
  logic [7:0] y;

  always_comb begin
    case (sel)
      3'b00x: y = 8'hd4;
      3'b0xz: y = 8'h17;
      3'bxz0: y = 8'h18;
      3'b01z: y = 8'h17;
      3'b10x: y = 8'hd4;
      3'bx1z: y = 8'hd4;
      3'bxx0: y = 8'hd4;
      3'bzxx: y = 8'h17;
      default: y = 8'h56;
    endcase
  end

  initial begin
    sel = 3'b000; #1; $display("A<%02h>", y);
    sel = 3'b001; #1; $display("B<%02h>", y);
    sel = 3'bzxx; #1; $display("C<%02h>", y);
    // CHECK: A<56>
    // CHECK: B<56>
    // CHECK: C<17>
    $finish;
  end
endmodule


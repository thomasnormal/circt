// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --mode interpret --top top 2>&1 | FileCheck %s

module top;
  logic [2:0] sel;
  logic [7:0] y;

  always_comb begin
    casez (sel)
      3'bzz1: y = 8'h01;
      default: y = 8'h02;
    endcase
  end

  initial begin
    sel = 3'b00x; #1; $display("A<%02h>", y);
    sel = 3'b00z; #1; $display("B<%02h>", y);
    // CHECK: A<02>
    // CHECK: B<01>
    $finish;
  end
endmodule

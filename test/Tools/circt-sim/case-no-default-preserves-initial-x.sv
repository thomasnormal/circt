// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --mode interpret --top top 2>&1 | FileCheck %s

module top;
  logic [2:0] sel = 3'b111;
  logic [7:0] y;

  always_comb begin
    case (sel)
      3'b001: y = 8'h01;
      3'b010: y = 8'h02;
    endcase
  end

  initial begin
    #1; $display("A<%02h>", y);
    sel = 3'b110; #1; $display("B<%02h>", y);
    // No case item matches, so y must retain its initial X state.
    // CHECK: A<xx>
    // CHECK: B<xx>
    $finish;
  end
endmodule

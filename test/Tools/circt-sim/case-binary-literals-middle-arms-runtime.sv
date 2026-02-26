// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --mode interpret --top tb 2>&1 | FileCheck %s

module mux4 (
  input  logic [1:0] sel,
  input  logic [7:0] a, b, c, d,
  output logic [7:0] y
);
  always_comb begin
    case (sel)
      2'b00: y = a;
      2'b01: y = b;
      2'b10: y = c;
      2'b11: y = d;
    endcase
  end
endmodule

module tb;
  logic [7:0] a = 8'hAA, b = 8'hBB, c = 8'hCC, d = 8'hDD;
  logic [1:0] sel;
  logic [7:0] y;

  mux4 dut(.sel, .a, .b, .c, .d, .y);

  initial begin
    sel = 0; #1 $display("sel=0 y=%0h", y);
    sel = 1; #1 $display("sel=1 y=%0h", y);
    sel = 2; #1 $display("sel=2 y=%0h", y);
    sel = 3; #1 $display("sel=3 y=%0h", y);
    $finish;
  end
endmodule

// CHECK: sel=0 y=aa
// CHECK: sel=1 y=bb
// CHECK: sel=2 y=cc
// CHECK: sel=3 y=dd

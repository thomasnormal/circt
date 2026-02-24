// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// CHECK: SVA_PASS: real local var incdec
// CHECK-NOT: SVA assertion failed

module top;
  bit clk = 0;
  real r = 2.5;

  always #5 clk = ~clk;

  property p_inc;
    real x;
    (1, x = r) ##1 (1, x++) ##1 (x == r + 1.0);
  endproperty

  property p_dec;
    real y;
    (1, y = r) ##1 (1, y--) ##1 (y == r - 1.0);
  endproperty

  a_inc: assert property (@(posedge clk) p_inc);
  a_dec: assert property (@(posedge clk) p_dec);

  initial begin
    repeat (8) @(posedge clk);
    $display("SVA_PASS: real local var incdec");
    $finish;
  end
endmodule

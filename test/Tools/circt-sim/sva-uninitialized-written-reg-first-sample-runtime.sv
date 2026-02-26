// RUN: circt-verilog %s --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top --max-time=200000000 2>&1 | FileCheck %s
// CHECK: SVA_PASS: uninitialized written reg starts as X
// CHECK-NOT: SVA assertion failed
//
// A 4-state reg without an explicit initializer must start as X, even if it is
// written later in a clocked process.

module top;
  logic clk = 0;
  reg c;
  logic seen = 0;

  always #5 clk = ~clk;

  always @(posedge clk) begin
    if (!seen) begin
      assert property (!$fell(c) && !$rose(c) && $stable(c));
      c <= 1'b1;
      seen <= 1'b1;
    end else begin
      $display("SVA_PASS: uninitialized written reg starts as X");
      $finish;
    end
  end
endmodule

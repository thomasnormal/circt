// RUN: circt-verilog %s --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top --max-time=200000000 2>&1 | FileCheck %s
// CHECK: SVA_PASS: first-sample stable semantics ok
// CHECK-NOT: SVA assertion failed
//
// First sampled value for `a` is unknown; with current value 0, `$stable(a)`
// should be false and `!$stable(a)` should pass on the first posedge.

module top;
  logic clk = 0;
  logic a = 0;
  logic seen = 0;

  always #5 clk = ~clk;

  always @(posedge clk) begin
    if (!seen) begin
      assert property (!$stable(a));
      seen <= 1;
    end else begin
      $display("SVA_PASS: first-sample stable semantics ok");
      $finish;
    end
  end
endmodule

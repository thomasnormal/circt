// RUN: circt-verilog %s --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top --max-time=500000000 2>&1 | FileCheck %s
// Test simple boolean concurrent assertion (no implication/delay).
// assert property (@(posedge clk) x != 0)
// x is always non-zero, so assertion always passes.

module top;
  reg clk;
  reg [7:0] x;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    x = 1;
    repeat(5) @(posedge clk);
    x = 42;
    repeat(5) @(posedge clk);

    // CHECK: SVA_PASS: boolean assertion ok
    $display("SVA_PASS: boolean assertion ok");
    $finish;
  end

  // CHECK-NOT: SVA assertion failed
  assert property (@(posedge clk) x != 0);
endmodule

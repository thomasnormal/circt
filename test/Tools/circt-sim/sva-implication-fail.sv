// RUN: circt-verilog %s --ir-llhd -o %t.mlir 2>&1
// RUN: circt-sim %t.mlir --top top --max-time=500000000 >%t.out 2>&1; FileCheck %s < %t.out
// Test that SVA assertion failure is detected when a |-> ##1 b is violated.
// a is high but b is NOT high on the next posedge.

module top;
  reg clk;
  reg a, b;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 0; b = 0;
    @(posedge clk);  // cycle 1
    a = 1; b = 0;    // a high at cycle 2
    @(posedge clk);  // cycle 2 (a sampled high)
    a = 0; b = 0;    // b NOT high at cycle 3 — violation!
    @(posedge clk);  // cycle 3 (b sampled low — assertion FAILS)
    a = 0; b = 0;
    @(posedge clk);  // cycle 4

    // CHECK: SVA assertion failed
    $display("SVA_DONE: stimulus complete");
    $finish;
  end

  assert property (@(posedge clk) a |-> ##1 b);
endmodule

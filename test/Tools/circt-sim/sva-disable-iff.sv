// RUN: circt-verilog %s --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top --max-time=500000000 2>&1 | FileCheck %s
// Test disable iff suppresses assertion failure during reset.
// Without disable iff, a |-> ##1 b would fail. But rst is high, so the
// assertion is vacuously true.

module top;
  reg clk, rst;
  reg a, b;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    rst = 1; a = 0; b = 0;
    @(posedge clk);  // cycle 1 (rst=1)
    a = 1; b = 0;    // a high, but rst is still 1
    @(posedge clk);  // cycle 2 (a sampled high, rst=1 -> disabled)
    a = 0; b = 0;    // b NOT high — would fail without disable iff
    @(posedge clk);  // cycle 3 (but rst=1, so assertion is disabled)

    // De-assert reset, verify assertion works normally
    rst = 0;
    a = 1; b = 0;
    @(posedge clk);  // cycle 4 (a high, rst=0)
    a = 0; b = 1;    // b high on next cycle — passes
    @(posedge clk);  // cycle 5
    a = 0; b = 0;
    @(posedge clk);  // cycle 6

    // CHECK: SVA_PASS: disable iff worked
    $display("SVA_PASS: disable iff worked");
    $finish;
  end

  // CHECK-NOT: SVA assertion failed
  assert property (@(posedge clk) disable iff (rst) a |-> ##1 b);
endmodule

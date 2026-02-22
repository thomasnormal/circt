// RUN: circt-verilog %s --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top --max-time=500000000 2>&1 | FileCheck %s
// Test concurrent SVA assertion with implication and ##1 delay.
// assert property (@(posedge clk) a |-> ##1 b)
// When a is high, b must be high on the next posedge.

module top;
  reg clk;
  reg a, b;

  // Generate clock: toggle every 5ns
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  // Drive stimulus: a=1 at cycle 2, b=1 at cycle 3 (satisfies a |-> ##1 b)
  initial begin
    a = 0; b = 0;
    @(posedge clk);  // cycle 1
    a = 1; b = 0;    // a high at cycle 2
    @(posedge clk);  // cycle 2 (a sampled high)
    a = 0; b = 1;    // b high at cycle 3 (one cycle after a)
    @(posedge clk);  // cycle 3 (b sampled high — assertion passes)
    a = 0; b = 0;
    @(posedge clk);  // cycle 4
    @(posedge clk);  // cycle 5

    // CHECK: SVA_PASS: no assertion failures
    $display("SVA_PASS: no assertion failures");
    $finish;
  end

  // This should NOT fire: a=1 at cycle 2 is followed by b=1 at cycle 3
  assert property (@(posedge clk) a |-> ##1 b);
endmodule

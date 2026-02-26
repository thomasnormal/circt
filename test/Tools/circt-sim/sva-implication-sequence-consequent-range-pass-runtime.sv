// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=200000000 2>&1 | FileCheck %s
// CHECK: SVA_PASS_RANGE_CONSEQUENT
// CHECK-NOT: SVA assertion failed at time

// Runtime semantics: bounded variable-length sequence consequents in
// implication should not fail immediately. This case should pass because
// `c` matches in the cycle after the later `b` match in ##[1:2].

module top;
  reg clk;
  reg trig, b, c;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    trig = 0;
    b = 0;
    c = 0;

    // S0 sample setup: trig=1, b=0, c=0.
    @(negedge clk);
    trig = 1;
    b = 0;
    c = 0;

    // S1: trig=0, b=1, c=0 (earliest b candidate).
    @(negedge clk);
    trig = 0;
    b = 1;
    c = 0;

    // S2: b=1, c=0 (later b candidate).
    @(negedge clk);
    b = 1;
    c = 0;

    // S3: b=0, c=1 (satisfies later candidate path).
    @(negedge clk);
    b = 0;
    c = 1;

    @(posedge clk);
    $display("SVA_PASS_RANGE_CONSEQUENT");
    $finish;
  end

  assert property (@(posedge clk)
      trig |-> ((##[1:2] b) ##1 c));
endmodule

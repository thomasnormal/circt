// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=200000000 >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// CHECK-NOT: SVA assertion failed at time 15000000 fs
// CHECK: SVA assertion failed at time 45000000 fs
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Runtime semantics: bounded variable-length sequence consequents in
// implication should fail when the bounded window closes without a match, not
// immediately at antecedent sampling.

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

    @(negedge clk);
    trig = 1;
    b = 0;
    c = 0;

    @(negedge clk);
    trig = 0;
    b = 1;
    c = 0;

    @(negedge clk);
    b = 1;
    c = 0;

    @(negedge clk);
    b = 0;
    c = 0;

    @(posedge clk);
    $finish;
  end

  assert property (@(posedge clk)
      trig |-> ((##[1:2] b) ##1 c));
endmodule

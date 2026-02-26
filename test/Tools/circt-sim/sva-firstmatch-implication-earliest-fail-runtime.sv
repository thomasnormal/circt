// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time 55000000 fs
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// first_match must bind the earliest delay-range hit. Here b is true at +1 and
// +3, but c is only true after the later hit; this must still fail.

module top;
  reg clk;
  reg a;
  reg b;
  reg c;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 0;
    b = 0;
    c = 0;

    @(posedge clk); // 5ns
    a = 1;

    @(posedge clk); // 15ns antecedent sampled
    a = 0;
    b = 1;

    @(posedge clk); // 25ns: earliest b (+1)
    b = 0;

    @(posedge clk); // 35ns: c should be required here (+2)
    b = 1;

    @(posedge clk); // 45ns: later b (+3)
    b = 0;
    c = 1;

    @(posedge clk); // 55ns
    $finish;
  end

  assert property (@(posedge clk) a |-> (first_match(##[1:3] b) ##1 c));
endmodule

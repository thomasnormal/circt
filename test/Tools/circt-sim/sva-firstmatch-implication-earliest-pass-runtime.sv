// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s
// CHECK: SVA_PASS_FIRSTMATCH_EARLIEST

// first_match should pass when c satisfies the earliest selected b hit.

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
    c = 1;

    @(posedge clk); // 35ns: c sampled for ##1 after earliest b
    c = 0;
    b = 1;

    @(posedge clk); // 45ns: later b (+3), should not be needed
    b = 0;

    @(posedge clk); // 55ns
    $display("SVA_PASS_FIRSTMATCH_EARLIEST");
    $finish;
  end

  assert property (@(posedge clk) a |-> (first_match(##[1:3] b) ##1 c));
endmodule

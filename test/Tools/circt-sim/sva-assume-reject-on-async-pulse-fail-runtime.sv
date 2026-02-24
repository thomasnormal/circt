// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s
// CHECK: SVA assumption failed at time
// CHECK: SVA assumption failure(s)
// CHECK: exit code 1

// Runtime semantics: async reject_on(c) on an assume must fail when c pulses
// high between sampled assertion clock edges.

module top;
  reg clk;
  reg a, b, c;

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b0;
    b = 1'b0;
    c = 1'b0;

    @(posedge clk); // cycle 1
    a = 1'b1;       // antecedent sampled at cycle 2
    b = 1'b1;       // consequent would pass without reject_on

    #1 c = 1'b1;    // async pulse between sampled edges
    #1 c = 1'b0;

    @(posedge clk); // cycle 2: reject_on should force assumption failure
    a = 1'b0;
    b = 1'b1;

    @(posedge clk); // cycle 3
    $finish;
  end

  assume property (@(posedge clk) reject_on(c) (a |-> ##1 b));
endmodule

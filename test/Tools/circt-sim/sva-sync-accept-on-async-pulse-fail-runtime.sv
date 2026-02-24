// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: not circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Runtime semantics: sync_accept_on(c) must ignore c pulses that only occur
// between sampled assertion clock edges.

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
    b = 1'b0;

    #1 c = 1'b1;    // async pulse between sampled edges
    #1 c = 1'b0;

    @(posedge clk); // cycle 2: no sampled abort, consequent fails
    a = 1'b0;
    b = 1'b0;

    @(posedge clk); // cycle 3
    $finish;
  end

  assert property (@(posedge clk) sync_accept_on(c) (a |-> ##1 b));
endmodule

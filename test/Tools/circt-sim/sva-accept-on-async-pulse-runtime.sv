// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s
// CHECK: SVA_PASS: accept_on observed async pulse between sampled edges
// CHECK-NOT: SVA assertion failed

// Runtime semantics: async accept_on(c) must vacuously pass when c pulses high
// between assertion clock edges while a consequent obligation is pending.

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
    a = 1'b1;       // antecedent sampled high at cycle 2
    b = 1'b0;

    #1 c = 1'b1;    // async pulse between sampled clock edges
    #1 c = 1'b0;

    @(posedge clk); // cycle 2 (consequent would fail without async abort)
    a = 1'b0;
    b = 1'b0;

    @(posedge clk); // cycle 3
    $display("SVA_PASS: accept_on observed async pulse between sampled edges");
    $finish;
  end

  assert property (@(posedge clk) accept_on(c) (a |-> ##1 b));
endmodule

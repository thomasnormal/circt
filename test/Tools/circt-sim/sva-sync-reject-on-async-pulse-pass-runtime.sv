// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=100000000 2>&1 | FileCheck %s
// CHECK: SVA_PASS: sync_reject_on ignored between-edge pulse
// CHECK-NOT: SVA assertion failed

// Runtime semantics: sync_reject_on(c) must ignore c pulses that only occur
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
    b = 1'b1;       // consequent satisfied at cycle 2

    #1 c = 1'b1;    // async pulse between sampled edges
    #1 c = 1'b0;

    @(posedge clk); // cycle 2: reject condition should not trigger
    a = 1'b0;
    b = 1'b1;

    @(posedge clk); // cycle 3
    $display("SVA_PASS: sync_reject_on ignored between-edge pulse");
    $finish;
  end

  assert property (@(posedge clk) sync_reject_on(c) (a |-> ##1 b));
endmodule

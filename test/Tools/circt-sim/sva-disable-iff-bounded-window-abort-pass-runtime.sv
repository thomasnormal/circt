// RUN: circt-verilog --no-uvm-auto-include %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=300000000 2>&1 | FileCheck %s
// CHECK: SVA_PASS: disable iff cleared bounded pending obligation
// CHECK-NOT: SVA assertion failed

// Runtime semantics: disable iff(c) must abort and clear pending implication
// obligations, not merely pause sample counting while c is active.

module top;
  reg clk;
  reg a, b, c;

  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

  initial begin
    a = 1'b0;
    b = 1'b0;
    c = 1'b0;

    @(posedge clk); // cycle 1
    a = 1'b1;       // antecedent sampled at cycle 2

    @(posedge clk); // cycle 2
    a = 1'b0;
    b = 1'b0;

    @(negedge clk);
    c = 1'b1;       // disable while ##[1:3] obligation is pending

    @(posedge clk); // cycle 3 (disabled sample)
    c = 1'b0;

    // Keep running long enough that a stale pending obligation would fail if
    // disable iff did not clear temporal state.
    repeat (6) @(posedge clk);

    $display("SVA_PASS: disable iff cleared bounded pending obligation");
    $finish;
  end

  assert property (@(posedge clk) disable iff (c) (a |-> ##[1:3] b));
endmodule
